/*
 * Copyright 2023-present the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.rag.advisor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

import org.jspecify.annotations.Nullable;
import reactor.core.scheduler.Scheduler;

import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseAdvisor;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.generation.augmentation.ContextualQueryAugmenter;
import org.springframework.ai.rag.generation.augmentation.QueryAugmenter;
import org.springframework.ai.rag.postretrieval.document.DocumentPostProcessor;
import org.springframework.ai.rag.preretrieval.query.expansion.QueryExpander;
import org.springframework.ai.rag.preretrieval.query.transformation.QueryTransformer;
import org.springframework.ai.rag.retrieval.join.ConcatenationDocumentJoiner;
import org.springframework.ai.rag.retrieval.join.DocumentJoiner;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.core.task.TaskExecutor;
import org.springframework.core.task.support.ContextPropagatingTaskDecorator;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.util.Assert;

/**
 * Advisor that implements common Retrieval Augmented Generation (RAG) flows using the
 * building blocks defined in the {@link org.springframework.ai.rag} package and following
 * the Modular RAG Architecture.
 *
 * <p>
 * 中文导读：在 ChatClient 的 advisor 链中编排 RAG，before 完成检索和提示词增强，
 * BaseAdvisor 随后继续调用下游链，after 再把使用过的文档附加到响应 metadata。
 * 本类没有直接调用 ChatModel 生成最终回答，也不负责把原始资料切块并导入向量库。
 *
 * <p>
 * 阅读 before 中的 0 到 7 步即可串起完整流程：构造 Query、串行变换、扩展多个查询、
 * 并行检索、合并文档、顺序后处理、增强原始问题、替换用户消息。
 * VectorStoreDocumentRetriever 是一种检索实现，ContextualQueryAugmenter 是默认增强实现；
 * 检索接口本身并不要求数据一定来自向量数据库。
 *
 * <p>
 * 检索使用变换和扩展后的 Query，但文档后处理与最终增强都接收 originalQuery。
 * 这样可以改善检索表达，同时保留用户真正提出的问题；返回文档的相关性不等于内容已被事实核验。
 * 是否允许无资料回答由 QueryAugmenter 决定，本 advisor 不会因空检索结果自动跳过模型调用。
 *
 * @author Christian Tzolov
 * @author Thomas Vitale
 * @since 1.0.0
 * @see <a href="http://export.arxiv.org/abs/2407.21059">arXiv:2407.21059</a>
 * @see <a href="https://export.arxiv.org/abs/2312.10997">arXiv:2312.10997</a>
 * @see <a href="https://export.arxiv.org/abs/2410.20878">arXiv:2410.20878</a>
 */
public final class RetrievalAugmentationAdvisor implements BaseAdvisor {

	public static final String DOCUMENT_CONTEXT = "rag_document_context";

	private final List<QueryTransformer> queryTransformers;

	private final @Nullable QueryExpander queryExpander;

	private final DocumentRetriever documentRetriever;

	private final DocumentJoiner documentJoiner;

	private final List<DocumentPostProcessor> documentPostProcessors;

	private final QueryAugmenter queryAugmenter;

	private final TaskExecutor taskExecutor;

	private final Scheduler scheduler;

	private final int order;

	private RetrievalAugmentationAdvisor(@Nullable List<QueryTransformer> queryTransformers,
			@Nullable QueryExpander queryExpander, DocumentRetriever documentRetriever,
			@Nullable DocumentJoiner documentJoiner, @Nullable List<DocumentPostProcessor> documentPostProcessors,
			@Nullable QueryAugmenter queryAugmenter, @Nullable TaskExecutor taskExecutor, @Nullable Scheduler scheduler,
			@Nullable Integer order) {
		Assert.notNull(documentRetriever, "documentRetriever cannot be null");
		Assert.noNullElements(queryTransformers, "queryTransformers cannot contain null elements");
		// 只强制要求检索器；未配置时跳过变换/扩展/后处理，合并器与增强器使用默认实现。
		this.queryTransformers = queryTransformers != null ? queryTransformers : List.of();
		this.queryExpander = queryExpander;
		this.documentRetriever = documentRetriever;
		this.documentJoiner = documentJoiner != null ? documentJoiner : new ConcatenationDocumentJoiner();
		this.documentPostProcessors = documentPostProcessors != null ? documentPostProcessors : List.of();
		this.queryAugmenter = queryAugmenter != null ? queryAugmenter : ContextualQueryAugmenter.builder().build();
		this.taskExecutor = taskExecutor != null ? taskExecutor : buildDefaultTaskExecutor();
		this.scheduler = scheduler != null ? scheduler : BaseAdvisor.DEFAULT_SCHEDULER;
		this.order = order != null ? order : 0;
	}

	public static Builder builder() {
		return new Builder();
	}

	@Override
	public ChatClientRequest before(ChatClientRequest chatClientRequest, @Nullable AdvisorChain advisorChain) {
		// 为这一轮复制上下文 Map，后续可加入检索文档而不直接修改传入请求的 Map 容器。
		Map<String, Object> context = new HashMap<>(chatClientRequest.context());

		// 0. Create a query from the user text, parameters, and conversation history.
		// history 来自当前 Prompt 的完整消息列表；本类不会自行读取 ChatMemory 或持久化会话。
		// Query.builder 最终要求 text 非空白，下面对 null 的空串替代并不代表接受空问题。
		String text = chatClientRequest.prompt().getUserMessage().getText();
		Query originalQuery = Query.builder()
			.text(Objects.requireNonNullElse(text, ""))
			.history(chatClientRequest.prompt().getInstructions())
			.context(context)
			.build();

		// 1. Transform original user query based on a chain of query transformers.
		// 每个 transformer 接收前一个的输出，顺序会影响语义；具体实现可能另行调用模型改写问题。
		Query transformedQuery = originalQuery;
		for (var queryTransformer : this.queryTransformers) {
			transformedQuery = queryTransformer.apply(transformedQuery);
		}

		// 2. Expand query into one or multiple queries.
		// 扩展器决定是否保留原查询以及生成几条查询，未配置时只检索 transformedQuery 一次。
		List<Query> expandedQueries = this.queryExpander != null ? this.queryExpander.expand(transformedQuery)
				: List.of(transformedQuery);

		// 3. Get similar documents for each query.
		// 先用 toList 提交完所有 future，再逐个 join；并发发生在不同查询的检索之间。
		// 当前 before 会等待检索结束，不是把 future 交给下游；此处没有超时、失败降级或取消其余任务。
		// 每个 Query 包装为一个检索结果列表，适配 Joiner 的“查询 -> 多个来源结果”输入结构。
		// toMap 没有重复键合并函数，扩展器输出相等的 Query 时会在收集阶段报错。
		Map<Query, List<List<Document>>> documentsForQuery = expandedQueries.stream()
			.map(query -> CompletableFuture.supplyAsync(() -> getDocumentsForQuery(query), this.taskExecutor))
			.toList()
			.stream()
			.map(CompletableFuture::join)
			.collect(Collectors.toMap(Map.Entry::getKey, entry -> List.of(entry.getValue())));

		// 4. Combine documents retrieved based on multiple queries and from multiple data
		// sources.
		// 默认 ConcatenationDocumentJoiner 按文档 id 去重，再按已有 score 降序排列。
		// 重复 id 保留遍历时遇到的第一份，不重算分数，也不保证选择该文档在多个查询中的最高分。
		// 此处没有统一的全局 topK，单次检索的 topK 不等于合并后上下文的总文档数上限。
		List<Document> documents = this.documentJoiner.join(documentsForQuery);

		// 5. Post-process the documents.
		// 可插入重排、压缩或过滤；处理器串行接力，并使用原始问题作为评估文档的依据。
		for (var documentPostProcessor : this.documentPostProcessors) {
			documents = documentPostProcessor.process(originalQuery, documents);
		}
		// 保留的是合并及后处理后的最终列表，既用于增强，也供响应阶段展示来源。
		context.put(DOCUMENT_CONTEXT, documents);

		// 6. Augment user query with the document contextual data.
		// 默认增强器将文档正文拼入提示词；这里刻意传 originalQuery，而非检索用的改写查询。
		Query augmentedQuery = this.queryAugmenter.augment(originalQuery, documents);

		// 7. Update ChatClientRequest with augmented prompt.
		// 只取 augmentedQuery.text() 更新用户消息；增强器返回的 history/context 不在这里重新合并。
		// Prompt 原有的其余消息及选项由 augmentUserMessage 保留，下游仍沿普通 advisor/model 链生成。
		return chatClientRequest.mutate()
			.prompt(chatClientRequest.prompt().augmentUserMessage(augmentedQuery.text()))
			.context(context)
			.build();
	}

	/**
	 * Processes a single query by routing it to document retrievers and collecting
	 * documents.
	 */
	private Map.Entry<Query, List<Document>> getDocumentsForQuery(Query query) {
		// 每个异步任务只调用当前配置的一个 DocumentRetriever；多数据源组合可由该实现内部承担。
		List<Document> documents = this.documentRetriever.retrieve(query);
		return Map.entry(query, documents);
	}

	@Override
	public ChatClientResponse after(ChatClientResponse chatClientResponse, @Nullable AdvisorChain advisorChain) {
		// 复制原模型响应后增加来源信息；文档放在 metadata 中，不会再被拼到模型生成的正文末尾。
		ChatResponse.Builder chatResponseBuilder;
		if (chatClientResponse.chatResponse() == null) {
			chatResponseBuilder = ChatResponse.builder();
		}
		else {
			chatResponseBuilder = ChatResponse.builder().from(chatClientResponse.chatResponse());
		}
		Object ctx = chatClientResponse.context().get(DOCUMENT_CONTEXT);
		if (ctx != null) {
			chatResponseBuilder.metadata(DOCUMENT_CONTEXT, ctx);
		}
		return ChatClientResponse.builder()
			.chatResponse(chatResponseBuilder.build())
			.context(chatClientResponse.context())
			.build();
	}

	@Override
	public Scheduler getScheduler() {
		// BaseAdvisor 的流式入口用它调度 before；它与并行检索使用的 taskExecutor 是两套执行机制。
		return this.scheduler;
	}

	@Override
	public int getOrder() {
		return this.order;
	}

	private static TaskExecutor buildDefaultTaskExecutor() {
		// 核心线程数 4、最大线程数 16；这里没有配置队列容量，实际扩容取决于执行器的排队策略。
		ThreadPoolTaskExecutor taskExecutor = new ThreadPoolTaskExecutor();
		taskExecutor.setThreadNamePrefix("ai-advisor-");
		taskExecutor.setCorePoolSize(4);
		taskExecutor.setMaxPoolSize(16);
		// 在线程切换时传播已注册的上下文，便于检索中的 Observation 接续调用链。
		// 这不等于把任意业务 ThreadLocal 自动复制到检索线程。
		taskExecutor.setTaskDecorator(new ContextPropagatingTaskDecorator());
		// 默认执行器在本类创建并初始化，但本类没有关闭钩子；需要统一生命周期时可注入应用托管执行器。
		taskExecutor.initialize();
		return taskExecutor;
	}

	public static final class Builder {

		private @Nullable List<QueryTransformer> queryTransformers;

		private @Nullable QueryExpander queryExpander;

		private @Nullable DocumentRetriever documentRetriever;

		private @Nullable DocumentJoiner documentJoiner;

		private @Nullable List<DocumentPostProcessor> documentPostProcessors;

		private @Nullable QueryAugmenter queryAugmenter;

		private @Nullable TaskExecutor taskExecutor;

		private @Nullable Scheduler scheduler;

		private @Nullable Integer order;

		private Builder() {
		}

		public Builder queryTransformers(List<QueryTransformer> queryTransformers) {
			Assert.noNullElements(queryTransformers, "queryTransformers cannot contain null elements");
			this.queryTransformers = queryTransformers;
			return this;
		}

		public Builder queryTransformers(QueryTransformer... queryTransformers) {
			Assert.notNull(queryTransformers, "queryTransformers cannot be null");
			Assert.noNullElements(queryTransformers, "queryTransformers cannot contain null elements");
			this.queryTransformers = Arrays.asList(queryTransformers);
			return this;
		}

		public Builder queryExpander(QueryExpander queryExpander) {
			this.queryExpander = queryExpander;
			return this;
		}

		public Builder documentRetriever(DocumentRetriever documentRetriever) {
			this.documentRetriever = documentRetriever;
			return this;
		}

		public Builder documentJoiner(DocumentJoiner documentJoiner) {
			this.documentJoiner = documentJoiner;
			return this;
		}

		public Builder documentPostProcessors(List<DocumentPostProcessor> documentPostProcessors) {
			Assert.noNullElements(documentPostProcessors, "documentPostProcessors cannot contain null elements");
			this.documentPostProcessors = documentPostProcessors;
			return this;
		}

		public Builder documentPostProcessors(DocumentPostProcessor... documentPostProcessors) {
			Assert.notNull(documentPostProcessors, "documentPostProcessors cannot be null");
			Assert.noNullElements(documentPostProcessors, "documentPostProcessors cannot contain null elements");
			this.documentPostProcessors = Arrays.asList(documentPostProcessors);
			return this;
		}

		public Builder queryAugmenter(QueryAugmenter queryAugmenter) {
			this.queryAugmenter = queryAugmenter;
			return this;
		}

		public Builder taskExecutor(TaskExecutor taskExecutor) {
			this.taskExecutor = taskExecutor;
			return this;
		}

		public Builder scheduler(Scheduler scheduler) {
			this.scheduler = scheduler;
			return this;
		}

		public Builder order(Integer order) {
			this.order = order;
			return this;
		}

		public RetrievalAugmentationAdvisor build() {
			Assert.state(this.documentRetriever != null, "documentRetriever cannot be null");
			return new RetrievalAugmentationAdvisor(this.queryTransformers, this.queryExpander, this.documentRetriever,
					this.documentJoiner, this.documentPostProcessors, this.queryAugmenter, this.taskExecutor,
					this.scheduler, this.order);
		}

	}

}
