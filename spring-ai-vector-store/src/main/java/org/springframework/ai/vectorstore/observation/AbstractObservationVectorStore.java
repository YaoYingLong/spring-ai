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

package org.springframework.ai.vectorstore.observation;

import java.util.List;

import io.micrometer.observation.ObservationRegistry;
import org.jspecify.annotations.Nullable;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.AbstractVectorStoreBuilder;
import org.springframework.ai.vectorstore.EmbeddedDocument;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;

/**
 * Abstract base class for {@link VectorStore} implementations that provides observation
 * capabilities.
 *
 * <p>
 * 中文导读：通过模板方法把统一观测与具体存储逻辑分开。
 * add、upsert、delete、similaritySearch 创建本次操作的 Context，再在 Micrometer Observation 内调用 doXxx。
 * 子类实现数据操作及观测属性；调用方应从公开业务入口进入，直接调用 doXxx 会绕开这一层包装。
 *
 * <p>
 * Context 只是请求、响应和数据库属性的载体，Convention 负责名称与标签，Registry 连接实际的观测处理器。
 * 是否产出指标或链路数据取决于注册的处理器，默认 builder 使用 NOOP Registry。
 * 默认 Convention 把操作名、数据库类型等作为低基数属性，查询文本和过滤表达式属于高基数属性。
 *
 * <p>
 * observe 包住的代码才属于该次观测的执行范围；文档类型校验和 Context 构建在包装之前发生。
 * 本类不增加重试、事务、批量拆分或统一异常兜底，存储失败仍沿调用链传播。
 *
 * @author Christian Tzolov
 * @author Soby Chacko
 * @since 1.0.0
 */
public abstract class AbstractObservationVectorStore implements VectorStore {

	private static final VectorStoreObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultVectorStoreObservationConvention();

	private final ObservationRegistry observationRegistry;

	private final @Nullable VectorStoreObservationConvention customObservationConvention;

	protected final EmbeddingModel embeddingModel;

	protected final BatchingStrategy batchingStrategy;

	private AbstractObservationVectorStore(EmbeddingModel embeddingModel, ObservationRegistry observationRegistry,
			@Nullable VectorStoreObservationConvention customObservationConvention, BatchingStrategy batchingStrategy) {
		this.embeddingModel = embeddingModel;
		this.observationRegistry = observationRegistry;
		this.customObservationConvention = customObservationConvention;
		this.batchingStrategy = batchingStrategy;
	}

	/**
	 * Creates a new AbstractObservationVectorStore instance with the specified builder
	 * settings. Initializes observation-related components and the embedding model.
	 * @param builder the builder containing configuration settings
	 */
	public AbstractObservationVectorStore(AbstractVectorStoreBuilder<?> builder) {
		this(builder.getEmbeddingModel(), builder.getObservationRegistry(), builder.getCustomObservationConvention(),
				builder.getBatchingStrategy());
	}

	/**
	 * Create a new {@link AbstractObservationVectorStore} instance.
	 * @param documents the documents to add
	 */
	@Override
	public void add(List<Document> documents) {
		// 文档载体支持媒体，但这条向量存储写入链路当前仅接受文本；空字符串文本仍可通过类型检查。
		validateNonTextDocuments(documents);
		VectorStoreObservationContext observationContext = this
			.createObservationContextBuilder(VectorStoreObservationContext.Operation.ADD.value())
			.build();

		// observe 负责生命周期和异常观测，lambda 内才执行具体存储的 embedding 与写入。
		VectorStoreObservationDocumentation.AI_VECTOR_STORE
			.observation(this.customObservationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> this.doAdd(documents));
	}

	/**
	 * Like {@link #add(List)}, this path accepts text documents only, but for a different
	 * reason: {@code add} is restricted because it invokes the text embedding model,
	 * while {@code upsert} never embeds and is restricted because no store can persist a
	 * row that carries no text. A document whose text is empty is fine, which is how a
	 * row points at content held outside the store.
	 */
	@Override
	public void upsert(List<EmbeddedDocument> entries) {
		// 已有 embedding 只免去向量生成，不免除当前存储层的文本内容约束。
		validateNonTextDocuments(entries.stream().map(EmbeddedDocument::document).toList());
		VectorStoreObservationContext observationContext = this
			.createObservationContextBuilder(VectorStoreObservationContext.Operation.UPSERT.value())
			.build();

		VectorStoreObservationDocumentation.AI_VECTOR_STORE
			.observation(this.customObservationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> this.doUpsert(entries));
	}

	private void validateNonTextDocuments(List<Document> documents) {
		// 这是内容类型检查，不是完整输入校验；null 列表和 null 元素交给后续实现处理。
		if (documents == null) {
			return;
		}
		for (Document document : documents) {
			if (document != null && !document.isText()) {
				throw new IllegalArgumentException(
						"Only text documents are supported for now. One of the documents contains non-text content.");
			}
		}
	}

	@Override
	public void delete(List<String> deleteDocIds) {

		// 两种删除入口都使用 DELETE 操作名，具体按 id 还是按过滤表达式删除由重载选择。
		VectorStoreObservationContext observationContext = this
			.createObservationContextBuilder(VectorStoreObservationContext.Operation.DELETE.value())
			.build();

		VectorStoreObservationDocumentation.AI_VECTOR_STORE
			.observation(this.customObservationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> this.doDelete(deleteDocIds));
	}

	@Override
	public void delete(Filter.Expression filterExpression) {
		VectorStoreObservationContext observationContext = this
			.createObservationContextBuilder(VectorStoreObservationContext.Operation.DELETE.value())
			.build();

		VectorStoreObservationDocumentation.AI_VECTOR_STORE
			.observation(this.customObservationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> this.doDelete(filterExpression));
	}

	@Override
	// Micrometer Observation#observe returns the value of the Supplier, which is never
	// null
	@SuppressWarnings("DataFlowIssue")
	public List<Document> similaritySearch(SearchRequest request) {

		// 在执行前关联 SearchRequest，供 Convention 读取 query、filter、topK 和 threshold。
		VectorStoreObservationContext searchObservationContext = this
			.createObservationContextBuilder(VectorStoreObservationContext.Operation.QUERY.value())
			.queryRequest(request)
			.build();

		return VectorStoreObservationDocumentation.AI_VECTOR_STORE
			.observation(this.customObservationConvention, DEFAULT_OBSERVATION_CONVENTION,
					() -> searchObservationContext, this.observationRegistry)
			.observe(() -> {
				var documents = this.doSimilaritySearch(request);
				// 只有检索成功后才记录响应，观测处理器可在结束时读取；不会改变返回文档或重新排序。
				searchObservationContext.setQueryResponse(documents);
				return documents;
			});
	}

	/**
	 * Perform the actual add operation.
	 * @param documents the documents to add
	 */
	public abstract void doAdd(List<Document> documents);

	/**
	 * Template method for concrete implementations to provide upsert logic using
	 * caller-supplied embeddings. The default implementation throws, so stores opt in
	 * independently.
	 * @param entries the documents and their pre-computed embeddings to upsert
	 */
	protected void doUpsert(List<EmbeddedDocument> entries) {
		// 父类提供观测入口不等于子类具备该能力，未覆写时仍明确失败。
		throw new UnsupportedOperationException(getName() + " does not support upsert");
	}

	/**
	 * Perform the actual delete operation.
	 * @param idList the list of document IDs to delete
	 */
	public abstract void doDelete(List<String> idList);

	/**
	 * Template method for concrete implementations to provide filter-based deletion
	 * logic.
	 * @param filterExpression Filter expression to identify documents to delete
	 */
	protected void doDelete(Filter.Expression filterExpression) {
		// this is temporary until we implement this method in all concrete vector stores,
		// at which point
		// this method will become an abstract method.
		throw new UnsupportedOperationException();
	}

	/**
	 * Perform the actual similarity search operation.
	 * @param request the search request
	 * @return the list of documents that match the query request conditions
	 */
	public abstract List<Document> doSimilaritySearch(SearchRequest request);

	/**
	 * Create a new {@link VectorStoreObservationContext.Builder} instance.
	 * @param operationName the operation name
	 * @return the observation context builder
	 */
	public abstract VectorStoreObservationContext.Builder createObservationContextBuilder(String operationName);

}
