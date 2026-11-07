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

package org.springframework.ai.vectorstore;

import java.util.Objects;

import org.jspecify.annotations.Nullable;

import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.ai.vectorstore.filter.FilterExpressionTextParser;
import org.springframework.util.Assert;

/**
 * Similarity search request. Use the {@link SearchRequest#builder()} to create the
 * instance of a {@link SearchRequest}.
 *
 * <p>
 * 中文导读：携带一次向量检索的查询文本、topK、相似度阈值和 metadata 过滤表达式。
 * 本类只构造请求，不生成查询向量、不读取文档，也不执行排序；这些工作由 VectorStore 完成。
 * 可先看 Builder 的参数校验，再看具体存储的 doSimilaritySearch 如何解释这些字段。
 *
 * <p>
 * topK 默认 4，阈值默认 0.0，过滤表达式默认 null。
 * topK 不保证一定返回足量文档，阈值也不是答案正确率；分数尺度与边界比较须结合后端实现理解。
 * 例如 SimpleVectorStore 对原始余弦分数执行大于等于阈值的比较，阈值 0.0 仍会排除负分。
 * 过滤在 Java 内存、数据库服务端还是其他阶段执行，也不是这个请求对象能够统一保证的。
 *
 * @author Christian Tzolov
 * @author Thomas Vitale
 * @author Ilayaperumal Gopinathan
 */
public class SearchRequest {

	/**
	 * Similarity threshold that accepts all search scores. A threshold value of 0.0 means
	 * any similarity is accepted or disable the similarity threshold filtering. A
	 * threshold value of 1.0 means an exact match is required.
	 */
	public static final double SIMILARITY_THRESHOLD_ACCEPT_ALL = 0.0;

	/**
	 * Default value for the top 'k' similar results to return.
	 * 默认查询返回相似度最高的top4
	 */
	public static final int DEFAULT_TOP_K = 4;

	/**
	 * Default value is empty string.
	 */
	private String query = "";

	private int topK = DEFAULT_TOP_K;

	// 相似度阀值，默认是0
	private double similarityThreshold = SIMILARITY_THRESHOLD_ACCEPT_ALL;

	// 根据文档的metadata过滤候选文档，只有满足条件的文档，才进入后面的相似度计算、排序和TopK选择。
	private Filter.@Nullable Expression filterExpression;

	/**
	 * Copy an existing {@link SearchRequest.Builder} instance.
	 * @param originalSearchRequest {@link SearchRequest} instance to copy.
	 * @return Returns new {@link SearchRequest.Builder} instance.
	 */
	public static Builder from(SearchRequest originalSearchRequest) {
		// 创建新的请求容器，适合 advisor 保留默认 topK/阈值、只替换本轮 query。
		// filterExpression 作为表达式对象直接传递，没有在此克隆整棵表达式树。
		return builder().query(originalSearchRequest.getQuery())
			.topK(originalSearchRequest.getTopK())
			.similarityThreshold(originalSearchRequest.getSimilarityThreshold())
			.filterExpression(originalSearchRequest.getFilterExpression());
	}

	public SearchRequest() {
	}

	protected SearchRequest(SearchRequest original) {
		this.query = original.query;
		this.topK = original.topK;
		this.similarityThreshold = original.similarityThreshold;
		this.filterExpression = original.filterExpression;
	}

	public String getQuery() {
		return this.query;
	}

	public int getTopK() {
		return this.topK;
	}

	public double getSimilarityThreshold() {
		return this.similarityThreshold;
	}

	public Filter.@Nullable Expression getFilterExpression() {
		return this.filterExpression;
	}

	public boolean hasFilterExpression() {
		return this.filterExpression != null;
	}

	@Override
	public String toString() {
		return "SearchRequest{" + "query='" + this.query + '\'' + ", topK=" + this.topK + ", similarityThreshold="
				+ this.similarityThreshold + ", filterExpression=" + this.filterExpression + '}';
	}

	@Override
	public boolean equals(@Nullable Object o) {
		if (this == o) {
			return true;
		}
		if (o == null || getClass() != o.getClass()) {
			return false;
		}
		SearchRequest that = (SearchRequest) o;
		return this.topK == that.topK && Double.compare(that.similarityThreshold, this.similarityThreshold) == 0
				&& Objects.equals(this.query, that.query)
				&& Objects.equals(this.filterExpression, that.filterExpression);
	}

	@Override
	public int hashCode() {
		return Objects.hash(this.query, this.topK, this.similarityThreshold, this.filterExpression);
	}

	/**
	 * Builder for creating the SearchRequest instance.
	 * @return the builder.
	 */
	public static Builder builder() {
		return new Builder();
	}

	/**
	 * SearchRequest Builder.
	 */
	public static final class Builder {

		private final SearchRequest searchRequest = new SearchRequest();

		/**
		 * @param query Text to use for embedding similarity comparison.
		 * @return this builder.
		 */
		public Builder query(String query) {
			// 允许空字符串，只拒绝 null；查询是否有业务意义、模型是否接受空输入由后续路径决定。
			Assert.notNull(query, "Query can not be null.");
			this.searchRequest.query = query;
			return this;
		}

		/**
		 * @param topK the top 'k' similar results to return.
		 * @return this builder.
		 */
		public Builder topK(int topK) {
			// 实际判断是 >= 0，所以 0 合法；SimpleVectorStore 的 limit(0) 会返回空列表。
			Assert.isTrue(topK >= 0, "TopK should be positive.");
			this.searchRequest.topK = topK;
			return this;
		}

		/**
		 * Similarity threshold score to filter the search response by. Only documents
		 * with similarity score equal or greater than the 'threshold' will be returned.
		 * Note that this is a post-processing step performed on the client not the server
		 * side. A threshold value of 0.0 means any similarity is accepted or disable the
		 * similarity threshold filtering. A threshold value of 1.0 means an exact match
		 * is required.
		 * <p>
		 * 中文边界：本方法只约束数值在 [0, 1]，不能统一指定阈值在哪里执行。
		 * 当前 SimpleVectorStore 在本地 Stream 过滤，PgVectorStore 会把阈值转换成 SQL 距离条件。
		 * 即使相似度达到 1，也不能据此断言原文完全相同或资料内容真实。
		 * @param threshold The lower bound of the similarity score.
		 * @return this builder.
		 */
		public Builder similarityThreshold(double threshold) {
			Assert.isTrue(threshold >= 0 && threshold <= 1, "Similarity threshold must be in [0,1] range.");
			this.searchRequest.similarityThreshold = threshold;
			return this;
		}

		/**
		 * Sets disables the similarity threshold by setting it to 0.0 - all results are
		 * accepted.
		 * @return this builder.
		 */
		public Builder similarityThresholdAll() {
			// 设置约定的默认值，没有额外携带“绕过比较”的布尔标志；实际效果仍由存储实现决定。
			this.searchRequest.similarityThreshold = 0.0;
			return this;
		}

		/**
		 * Retrieves documents by query embedding similarity and matching the filters.
		 * Value of 'null' means that no metadata filters will be applied to the search.
		 *
		 * For example if the {@link Document#getMetadata()} schema is:
		 *
		 * <pre>{@code
		 * &#123;
		 * "country": <Text>,
		 * "city": <Text>,
		 * "year": <Number>,
		 * "price": <Decimal>,
		 * "isActive": <Boolean>
		 * &#125;
		 * }</pre>
		 *
		 * you can constrain the search result to only UK countries with isActive=true and
		 * year equal or greater 2020. You can build this such metadata filter
		 * programmatically like this:
		 *
		 * <pre>{@code
		 * var exp = new Filter.Expression(AND,
		 * 		new Expression(EQ, new Key("country"), new Value("UK")),
		 * 		new Expression(AND,
		 * 				new Expression(GTE, new Key("year"), new Value(2020)),
		 * 				new Expression(EQ, new Key("isActive"), new Value(true))));
		 * }</pre>
		 *
		 * The {@link Filter.Expression} is portable across all vector stores.<br/>
		 *
		 *
		 * The {@link FilterExpressionBuilder} is a DSL creating expressions
		 * programmatically:
		 *
		 * <pre>{@code
		 * var b = new FilterExpressionBuilder();
		 * var exp = b.and(
		 * 		b.eq("country", "UK"),
		 * 		b.and(
		 * 			b.gte("year", 2020),
		 * 			b.eq("isActive", true)));
		 * }</pre>
		 *
		 * The {@link FilterExpressionTextParser} converts textual, SQL like filter
		 * expression language into {@link Filter.Expression}:
		 *
		 * <pre>{@code
		 * var parser = new FilterExpressionTextParser();
		 * var exp = parser.parse("country == 'UK' && isActive == true && year >=2020");
		 * }</pre>
		 * @param expression {@link Filter.Expression} instance used to define the
		 * metadata filter criteria. The 'null' value stands for no expression filters.
		 * @return this builder.
		 */
		public Builder filterExpression(Filter.@Nullable Expression expression) {
			// 保存可移植的表达式树；各存储可将其转成原生查询，或像 SimpleVectorStore 一样本地求值。
			this.searchRequest.filterExpression = expression;
			return this;
		}

		/**
		 * Document metadata filter expression. For example if your
		 * {@link Document#getMetadata()} has a schema like:
		 *
		 * <pre>{@code
		 * &#123;
		 * "country": <Text>,
		 * "city": <Text>,
		 * "year": <Number>,
		 * "price": <Decimal>,
		 * "isActive": <Boolean>
		 * &#125;
		 * }</pre>
		 *
		 * then you can constrain the search result with metadata filter expressions like:
		 *
		 * <pre>{@code
		 * country == 'UK' && year >= 2020 && isActive == true
		 * Or
		 * country == 'BG' && (city NOT IN ['Sofia', 'Plovdiv'] || price < 134.34)
		 * }</pre>
		 *
		 * This ensures that the response contains only embeddings that match the
		 * specified filer criteria. <br/>
		 *
		 * The declarative, SQL like, filter syntax is portable across all vector stores
		 * supporting the filter search feature.<br/>
		 *
		 * The {@link FilterExpressionTextParser} is used to convert the text filter
		 * expression into {@link Filter.Expression}.
		 * @param textExpression declarative, portable, SQL like, metadata filter syntax.
		 * The 'null' value stands for no expression filters.
		 * @return this.builder
		 */
		public Builder filterExpression(@Nullable String textExpression) {
			// 非 null 字符串立即解析，语法错误在构造请求时暴露；null 表示不设置 metadata 条件。
			this.searchRequest.filterExpression = (textExpression != null)
					? new FilterExpressionTextParser().parse(textExpression) : null;
			return this;
		}

		public SearchRequest build() {
			// 返回 Builder 持有的同一实例；build 后继续调用该 Builder 的 setter 会影响已返回的请求。
			return this.searchRequest;
		}

	}

}
