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

package org.springframework.ai.rag.retrieval.search;

import java.util.List;
import java.util.function.Supplier;

import org.jspecify.annotations.Nullable;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionTextParser;
import org.springframework.util.Assert;
import org.springframework.util.StringUtils;

/**
 * Retrieves documents from a vector store that are semantically similar to the input
 * query. It supports filtering based on metadata, similarity threshold, and top-k
 * results.
 *
 * <p>
 * Example usage: <pre>{@code
 * VectorStoreDocumentRetriever retriever = VectorStoreDocumentRetriever.builder()
 *     .vectorStore(vectorStore)
 *     .similarityThreshold(0.73)
 *     .topK(5)
 *     .filterExpression(filterExpression)
 *     .build();
 * List<Document> documents = retriever.retrieve(new Query("example query"));
 * }</pre>
 *
 * <p>
 * The {@link #FILTER_EXPRESSION} context key can be used to provide a filter expression
 * for a specific query. This key accepts either a string representation of a filter
 * expression or a {@link Filter.Expression} object directly.
 *
 * <p>
 * 中文导读：把模块化 RAG 的 Query 适配为 VectorStore 的 SearchRequest。
 * retrieve 只读取 query.text 和约定的过滤上下文，不会把 history 自动拼成检索文本。
 * 查询向量生成、metadata 过滤、评分与 topK 选择继续由具体向量存储完成。
 *
 * <p>
 * 调用从 RetrievalAugmentationAdvisor 进入 retrieve，经过 computeRequestFilterExpression，
 * 最后进入 similaritySearch。
 * 默认 topK 为 4、相似度阈值为 0.0；每次 retrieve 都创建独立请求。
 * 请求上下文提供的非空过滤条件会替换默认条件，而不会自动与默认条件做 AND 合并。
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
public final class VectorStoreDocumentRetriever implements DocumentRetriever {

	public static final String FILTER_EXPRESSION = "vector_store_filter_expression";

	private final VectorStore vectorStore;

	private final Double similarityThreshold;

	private final Integer topK;

	// Supplier to allow for lazy evaluation of the filter expression,
	// which may depend on the execution content. For example, you may want to
	// filter dynamically based on the current user's identity or tenant ID.
	private final Supplier<Filter.Expression> filterExpression;

	public VectorStoreDocumentRetriever(VectorStore vectorStore, @Nullable Double similarityThreshold,
			@Nullable Integer topK, @Nullable Supplier<Filter.Expression> filterExpression) {
		Assert.notNull(vectorStore, "vectorStore cannot be null");
		Assert.isTrue(similarityThreshold == null || similarityThreshold >= 0.0,
				"similarityThreshold must be equal to or greater than 0.0");
		// 此处 topK 要求大于 0，比 SearchRequest 允许 0 的边界更严格。
		// threshold 的上限 1 则在 retrieve 构造 SearchRequest 时才由其 Builder 检查。
		Assert.isTrue(topK == null || topK > 0, "topK must be greater than 0");
		this.vectorStore = vectorStore;
		this.similarityThreshold = similarityThreshold != null ? similarityThreshold
				: SearchRequest.SIMILARITY_THRESHOLD_ACCEPT_ALL;
		this.topK = topK != null ? topK : SearchRequest.DEFAULT_TOP_K;
		this.filterExpression = filterExpression != null ? filterExpression : () -> null;
	}

	@Override
	public List<Document> retrieve(Query query) {
		Assert.notNull(query, "query cannot be null");
		// 在实际执行 retrieve 的线程上解析本轮条件，必要时调用默认 Supplier。
		// 当 advisor 并行检索时，这里位于 taskExecutor 线程，不能假定仍在原请求线程。
		var requestFilterExpression = computeRequestFilterExpression(query);
		var searchRequest = SearchRequest.builder()
			.query(query.text())
			.filterExpression(requestFilterExpression)
			.similarityThreshold(this.similarityThreshold)
			.topK(this.topK)
			.build();
		// 走存储的公开入口，以保留 AbstractObservationVectorStore 提供的观测包装。
		// 不在这里重排或截断返回值，避免把后端实现差异隐藏成另一套检索语义。
		return this.vectorStore.similaritySearch(searchRequest);
	}

	/**
	 * Computes the filter expression to use for the current request.
	 * <p>
	 * The filter expression can be provided in the query context using the
	 * {@link #FILTER_EXPRESSION} key. This key accepts either a string representation of
	 * a filter expression or a {@link Filter.Expression} object directly.
	 * <p>
	 * If no filter expression is provided in the context, the default filter expression
	 * configured for this retriever is used.
	 * @param query the query containing potential context with filter expression
	 * @return the filter expression to use for the request
	 */
	private Filter.Expression computeRequestFilterExpression(Query query) {
		// 优先级：上下文中的 Expression、可解析的非空文本、构建时配置的 Supplier。
		var contextFilterExpression = query.context().get(FILTER_EXPRESSION);
		if (contextFilterExpression != null) {
			if (contextFilterExpression instanceof Filter.Expression) {
				return (Filter.Expression) contextFilterExpression;
			}
			else if (StringUtils.hasText(contextFilterExpression.toString())) {
				// 其他对象也先调用 toString 再解析，语法错误向外传播，不会悄悄退回默认条件。
				return new FilterExpressionTextParser().parse(contextFilterExpression.toString());
			}
		}
		// null 或空白上下文走默认条件；因此传空字符串并不能清除已配置的默认过滤。
		// Supplier 每次需要默认条件时执行，可提供动态值；其线程上下文由调用方负责安排。
		return this.filterExpression.get();
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Builder for {@link VectorStoreDocumentRetriever}.
	 */
	public static final class Builder {

		private @Nullable VectorStore vectorStore;

		private @Nullable Double similarityThreshold;

		private @Nullable Integer topK;

		private @Nullable Supplier<Filter.Expression> filterExpression;

		private Builder() {
		}

		public Builder vectorStore(VectorStore vectorStore) {
			this.vectorStore = vectorStore;
			return this;
		}

		public Builder similarityThreshold(Double similarityThreshold) {
			this.similarityThreshold = similarityThreshold;
			return this;
		}

		public Builder topK(Integer topK) {
			this.topK = topK;
			return this;
		}

		public Builder filterExpression(Filter.Expression filterExpression) {
			this.filterExpression = () -> filterExpression;
			return this;
		}

		public Builder filterExpression(Supplier<Filter.Expression> filterExpression) {
			this.filterExpression = filterExpression;
			return this;
		}

		public VectorStoreDocumentRetriever build() {
			Assert.state(this.vectorStore != null, "vectorStore cannot be null");
			return new VectorStoreDocumentRetriever(this.vectorStore, this.similarityThreshold, this.topK,
					this.filterExpression);
		}

	}

}
