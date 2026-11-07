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

package org.springframework.ai.chat.client.advisor.vectorstore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import org.jspecify.annotations.Nullable;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseAdvisor;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionTextParser;
import org.springframework.util.Assert;
import org.springframework.util.StringUtils;

/**
 * Context for the question is retrieved from a Vector Store and added to the prompt's
 * user text.
 *
 * <p>
 * 中文导读：通过一次向量检索与一次用户消息增强实现基础 RAG。
 * BaseAdvisor 负责 before、下游模型调用、after 的公共流程，本类只提供检索、拼接和来源回传。
 * 与模块化 RetrievalAugmentationAdvisor 相比，这里没有查询变换、扩展、文档合并或后处理插槽。
 *
 * <p>
 * 阅读顺序为 before 中的 SearchRequest 构造、similaritySearch、PromptTemplate.render，最后看 after。
 * 检索使用当前用户文本，默认模板还提示模型结合已有历史回答；本类不会自行读取会话存储。
 * 相似文档只是生成依据，默认提示词要求资料不足时拒答，并不在代码层保证模型回答正确。
 *
 * @author Christian Tzolov
 * @author Timo Salm
 * @author Ilayaperumal Gopinathan
 * @author Thomas Vitale
 * @author Yanming Zhou
 * @since 1.0.0
 */
public class QuestionAnswerAdvisor implements BaseAdvisor {

	public static final String RETRIEVED_DOCUMENTS = "qa_retrieved_documents";

	public static final String FILTER_EXPRESSION = "qa_filter_expression";

	private static final PromptTemplate DEFAULT_PROMPT_TEMPLATE = new PromptTemplate("""
			{query}

			Context information is below, surrounded by ---------------------

			---------------------
			{question_answer_context}
			---------------------

			Given the context and provided history information and not prior knowledge,
			reply to the user comment. If the answer is not in the context, inform
			the user that you can't answer the question.
			""");

	private static final int DEFAULT_ORDER = 0;

	private final VectorStore vectorStore;

	private final PromptTemplate promptTemplate;

	private final SearchRequest searchRequest;

	private final Scheduler scheduler;

	private final int order;

	QuestionAnswerAdvisor(VectorStore vectorStore, SearchRequest searchRequest, @Nullable PromptTemplate promptTemplate,
			@Nullable Scheduler scheduler, int order) {
		Assert.notNull(vectorStore, "vectorStore cannot be null");
		Assert.notNull(searchRequest, "searchRequest cannot be null");

		this.vectorStore = vectorStore;
		this.searchRequest = searchRequest;
		this.promptTemplate = promptTemplate != null ? promptTemplate : DEFAULT_PROMPT_TEMPLATE;
		this.scheduler = scheduler != null ? scheduler : BaseAdvisor.DEFAULT_SCHEDULER;
		this.order = order;
	}

	public static Builder builder(VectorStore vectorStore) {
		return new Builder(vectorStore);
	}

	@Override
	public int getOrder() {
		return this.order;
	}

	@Override
	public ChatClientRequest before(ChatClientRequest chatClientRequest, AdvisorChain advisorChain) {
		// 1. Search for similar documents in the vector store.
		// 每轮复制模板请求，再以当前用户文本替换 query，保留已配置的 topK、阈值与默认过滤条件。
		var searchRequestBuilder = SearchRequest.from(this.searchRequest)
			.query(Objects.requireNonNullElse(chatClientRequest.prompt().getUserMessage().getText(), ""));

		// 动态过滤条件优先于固定条件，是替换关系，不自动将两者做 AND 合并。
		var filterExpr = doGetFilterExpression(chatClientRequest.context());
		if (filterExpr != null) {
			searchRequestBuilder.filterExpression(filterExpr);
		}

		var searchRequestToUse = searchRequestBuilder.build();

		// 使用公开存储入口，向量生成、相似度算法及 Observation 包装由具体存储链路负责。
		List<Document> documents = this.vectorStore.similaritySearch(searchRequestToUse);

		// 2. Create the context from the documents.
		Map<String, Object> context = new HashMap<>(chatClientRequest.context());
		context.put(RETRIEVED_DOCUMENTS, documents);

		// 模型上下文只拼正文，不自动加入来源 id、metadata 或 score；完整 Document 另存于 advisor context。
		String documentContext = documents.stream()
			.map(Document::getText)
			.collect(Collectors.joining(System.lineSeparator()));

		// 3. Augment the user prompt with the document context.
		// 结果为空时 documentContext 是空串，仍照常渲染并继续下游调用；不存在代码层的空检索短路。
		UserMessage userMessage = chatClientRequest.prompt().getUserMessage();
		String augmentedUserText = this.promptTemplate
			.render(Map.of("query", userMessage.getText(), "question_answer_context", documentContext));

		// 4. Update ChatClientRequest with augmented prompt.
		// 只替换当前用户消息文本，Prompt 中的其他消息和模型选项继续保留。
		return chatClientRequest.mutate()
			.prompt(chatClientRequest.prompt().augmentUserMessage(augmentedUserText))
			.context(context)
			.build();
	}

	@Override
	public ChatClientResponse after(ChatClientResponse chatClientResponse, AdvisorChain advisorChain) {
		// 将实际召回列表回传到 ChatResponse.metadata，便于调用方展示资料来源或排查召回质量。
		// 这里不改模型回答正文；流式场景的 after 调用时机由 BaseAdvisor 的结束标记判断控制。
		ChatResponse.Builder chatResponseBuilder = ChatResponse.builder();
		if (chatClientResponse.chatResponse() != null) {
			chatResponseBuilder.from(chatClientResponse.chatResponse());
		}
		if (chatClientResponse.context().get(RETRIEVED_DOCUMENTS) != null) {
			chatResponseBuilder.metadata(RETRIEVED_DOCUMENTS, chatClientResponse.context().get(RETRIEVED_DOCUMENTS));
		}
		return ChatClientResponse.builder()
			.chatResponse(chatResponseBuilder.build())
			.context(chatClientResponse.context())
			.build();
	}

	protected Filter.@Nullable Expression doGetFilterExpression(Map<String, @Nullable Object> context) {
		// 未设置或空白时回退到固定请求的过滤条件；有值时按文本语法解析，不直接传递 Expression 对象。
		Object ctxFilterExpr = context.get(FILTER_EXPRESSION);
		if (ctxFilterExpr == null || !StringUtils.hasText(ctxFilterExpr.toString())) {
			return this.searchRequest.getFilterExpression();
		}
		return new FilterExpressionTextParser().parse(ctxFilterExpr.toString());
	}

	@Override
	public Scheduler getScheduler() {
		// BaseAdvisor 的流式入口在此 scheduler 上执行 before；同步 adviseCall 仍在调用线程执行。
		return this.scheduler;
	}

	public static final class Builder {

		private final VectorStore vectorStore;

		private SearchRequest searchRequest = SearchRequest.builder().build();

		private @Nullable PromptTemplate promptTemplate;

		private @Nullable Scheduler scheduler;

		private int order = DEFAULT_ORDER;

		private Builder(VectorStore vectorStore) {
			Assert.notNull(vectorStore, "The vectorStore must not be null!");
			this.vectorStore = vectorStore;
		}

		public Builder promptTemplate(PromptTemplate promptTemplate) {
			Assert.notNull(promptTemplate, "promptTemplate cannot be null");
			this.promptTemplate = promptTemplate;
			return this;
		}

		public Builder searchRequest(SearchRequest searchRequest) {
			Assert.notNull(searchRequest, "The searchRequest must not be null!");
			this.searchRequest = searchRequest;
			return this;
		}

		public Builder protectFromBlocking(boolean protectFromBlocking) {
			// 影响流式链路的调度选择，不会把底层同步 VectorStore API 转换为异步 API。
			this.scheduler = protectFromBlocking ? BaseAdvisor.DEFAULT_SCHEDULER : Schedulers.immediate();
			return this;
		}

		public Builder scheduler(Scheduler scheduler) {
			this.scheduler = scheduler;
			return this;
		}

		public Builder order(int order) {
			this.order = order;
			return this;
		}

		public QuestionAnswerAdvisor build() {
			return new QuestionAnswerAdvisor(this.vectorStore, this.searchRequest, this.promptTemplate, this.scheduler,
					this.order);
		}

	}

}
