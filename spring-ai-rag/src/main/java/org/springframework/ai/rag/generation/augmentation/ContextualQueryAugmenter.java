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

package org.springframework.ai.rag.generation.augmentation;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jspecify.annotations.Nullable;

import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.util.PromptAssert;
import org.springframework.util.Assert;

/**
 * Augments the user query with contextual data from the content of the provided
 * documents.
 *
 * <p>
 * Example usage: <pre>{@code
 * QueryAugmenter augmenter = ContextualQueryAugmenter.builder()
 *    .allowEmptyContext(false)
 *    .build();
 * Query augmentedQuery = augmenter.augment(query, documents);
 * }</pre>
 *
 * <p>
 * 中文导读：RAG 检索完成后的提示词组装器，接收问题与最终文档列表，输出增强后的 Query。
 * 它既不检索文档，也不调用聊天模型；RetrievalAugmentationAdvisor 只把返回 Query 的 text 写回用户消息。
 * 可沿 augment 的文档非空分支和 augmentQueryWhenEmptyContext 的空结果分支阅读。
 *
 * <p>
 * 默认只把文档正文按换行连接，metadata、id 和 score 不会自动出现在模型上下文中。
 * 如需给模型页码、来源或引用标签，应通过 documentFormatter 明确组织这些信息。
 * 这里没有 token 预算控制或内容真实性校验，召回过多资料的压缩、重排可放到前置后处理器中。
 *
 * <p>
 * allowEmptyContext 默认 false 表示生成一段要求模型礼貌拒答的提示词，仍会继续调用下游模型。
 * 这不是 Java 层强制返回拒答结果；模型是否遵循“仅根据资料回答”的要求仍需另行评估。
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
public final class ContextualQueryAugmenter implements QueryAugmenter {

	private static final Log logger = LogFactory.getLog(ContextualQueryAugmenter.class);

	private static final PromptTemplate DEFAULT_PROMPT_TEMPLATE = new PromptTemplate("""
			Context information is below.

			---------------------
			{context}
			---------------------

			Given the context information and no prior knowledge, answer the query.

			Follow these rules:

			1. If the answer is not in the context, just say that you don't know.
			2. Avoid statements like "Based on the context..." or "The provided information...".

			Query: {query}

			Answer:
			""");

	private static final PromptTemplate DEFAULT_EMPTY_CONTEXT_PROMPT_TEMPLATE = new PromptTemplate("""
			The user query is outside your knowledge base.
			Politely inform the user that you can't answer it.
			""");

	private static final boolean DEFAULT_ALLOW_EMPTY_CONTEXT = false;

	/**
	 * Default document formatter that just joins document text with newlines
	 */
	private static final Function<List<Document>, String> DEFAULT_DOCUMENT_FORMATTER = documents -> documents.stream()
		.map(Document::getText)
		.collect(Collectors.joining(System.lineSeparator()));

	private final PromptTemplate promptTemplate;

	private final PromptTemplate emptyContextPromptTemplate;

	private final boolean allowEmptyContext;

	private final Function<List<Document>, String> documentFormatter;

	public ContextualQueryAugmenter(@Nullable PromptTemplate promptTemplate,
			@Nullable PromptTemplate emptyContextPromptTemplate, @Nullable Boolean allowEmptyContext,
			@Nullable Function<List<Document>, String> documentFormatter) {
		this.promptTemplate = promptTemplate != null ? promptTemplate : DEFAULT_PROMPT_TEMPLATE;
		this.emptyContextPromptTemplate = emptyContextPromptTemplate != null ? emptyContextPromptTemplate
				: DEFAULT_EMPTY_CONTEXT_PROMPT_TEMPLATE;
		this.allowEmptyContext = allowEmptyContext != null ? allowEmptyContext : DEFAULT_ALLOW_EMPTY_CONTEXT;
		this.documentFormatter = documentFormatter != null ? documentFormatter : DEFAULT_DOCUMENT_FORMATTER;
		// 非空上下文模板必须包含 query 与 context 占位符，确保两个输入都有明确插入位置。
		PromptAssert.templateHasRequiredPlaceholders(this.promptTemplate, "query", "context");
	}

	@Override
	public Query augment(Query query, List<Document> documents) {
		Assert.notNull(query, "query cannot be null");
		Assert.notNull(documents, "documents cannot be null");

		logger.debug("Augmenting query with contextual data");

		if (documents.isEmpty()) {
			// 判定依据是文档列表为空，不是拼接后的正文为空；空正文文档仍会走普通增强分支。
			return augmentQueryWhenEmptyContext(query);
		}

		// 1. Collect content from documents.
		// 保留传入列表的顺序，由上游决定哪些文档更靠前；formatter 负责资料如何呈现。
		String documentContext = this.documentFormatter.apply(documents);

		// 2. Define prompt parameters.
		Map<String, Object> promptParameters = Map.of("query", query.text(), "context", documentContext);

		// 3. Augment user prompt with document context.
		// 重新构造的 Query 只有文本，history/context 使用 Query(String) 的空默认值。
		// advisor 只读取该文本，所以原 Prompt 的历史消息不会因此被移除。
		return new Query(this.promptTemplate.render(promptParameters));
	}

	private Query augmentQueryWhenEmptyContext(Query query) {
		if (this.allowEmptyContext) {
			// 显式允许无资料时返回原 Query，模型可按原问题继续生成，没有新增检索上下文。
			logger.debug("Empty context is allowed. Returning the original query.");
			return query;
		}
		logger.debug("Empty context is not allowed. Returning a specific query for empty context.");
		// 默认替换为“知识库范围之外”的指令，原问题不再插入该模板，也没有抛异常或构造 ChatResponse。
		return new Query(this.emptyContextPromptTemplate.render());
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private @Nullable PromptTemplate promptTemplate;

		private @Nullable PromptTemplate emptyContextPromptTemplate;

		private @Nullable Boolean allowEmptyContext;

		private @Nullable Function<List<Document>, String> documentFormatter;

		public Builder promptTemplate(PromptTemplate promptTemplate) {
			this.promptTemplate = promptTemplate;
			return this;
		}

		public Builder emptyContextPromptTemplate(PromptTemplate emptyContextPromptTemplate) {
			this.emptyContextPromptTemplate = emptyContextPromptTemplate;
			return this;
		}

		public Builder allowEmptyContext(Boolean allowEmptyContext) {
			this.allowEmptyContext = allowEmptyContext;
			return this;
		}

		public Builder documentFormatter(Function<List<Document>, String> documentFormatter) {
			this.documentFormatter = documentFormatter;
			return this;
		}

		public ContextualQueryAugmenter build() {
			return new ContextualQueryAugmenter(this.promptTemplate, this.emptyContextPromptTemplate,
					this.allowEmptyContext, this.documentFormatter);
		}

	}

}
