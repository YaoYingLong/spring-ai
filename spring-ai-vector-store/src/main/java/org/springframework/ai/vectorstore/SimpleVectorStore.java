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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jspecify.annotations.Nullable;
import tools.jackson.core.JacksonException;
import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.ObjectWriter;
import tools.jackson.databind.json.JsonMapper;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.observation.conventions.VectorStoreProvider;
import org.springframework.ai.observation.conventions.VectorStoreSimilarityMetric;
import org.springframework.ai.util.JacksonUtils;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.observation.AbstractObservationVectorStore;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationContext;
import org.springframework.core.io.Resource;

/**
 * A simple, in-memory implementation of the <a href=
 * "https://docs.spring.io/spring-ai/reference/api/vectordbs.html#_understanding_vectors">VectorStore</a>
 * interface.
 *
 * <p>
 * Uses a {@link java.util.concurrent.ConcurrentHashMap} to store vectors and their
 * associated metadata. Map keys are document IDs; values are
 * {@link SimpleVectorStoreContent} instances that encapsulate each document's text,
 * metadata, and embedding vector.
 *
 * <p>
 * Similarity search is performed using cosine similarity over all stored vectors. Filter
 * expressions on document metadata are evaluated via
 * {@link SimpleVectorStoreFilterExpressionEvaluator}.
 *
 * <p>
 * The store can be persisted to and restored from a JSON file via the
 * {@link #save(java.io.File)} and {@link #load(java.io.File)} /
 * {@link #load(org.springframework.core.io.Resource)} methods.
 *
 * <p>
 * <b>NOTE</b>: This implementation is not designed for production use and should only be
 * used for testing or demonstration purposes.
 *
 * <p>
 * 中文导读：用内存 Map 展示向量检索完整机制，适合沿 add、doAdd、similaritySearch、
 * doSimilaritySearch 阅读。公开入口的观测包装在父类，本类实现向量生成、保存、过滤和余弦排序。
 * 它没有向量索引，查询会遍历存储记录，并对通过 metadata 条件的候选逐个计算分数。
 *
 * <p>
 * 文档写入使用 EmbeddingModel.embed(Document)，查询使用同一模型对象的 embed(String)。
 * 两端文本的格式化入口不同，加载的历史向量也不会校验模型身份；应由使用方保持向量空间兼容。
 * 返回分数是与当前查询的余弦相似度，并不表示文档真实性、业务价值或回答正确概率。
 *
 * <p>
 * save/load 是显式 JSON 快照操作，不是自动持久化或事务日志。
 * 当前类未覆写 doUpsert，因此即使 add 使用 Map.put 覆盖同 id，公开 upsert 仍属于未支持能力。
 *
 * @author Raphael Yu
 * @author Dingmeng Xue
 * @author Mark Pollack
 * @author Christian Tzolov
 * @author Sebastien Deleuze
 * @author Ilayaperumal Gopinathan
 * @author Thomas Vitale
 * @author Jemin Huh
 * @author David Yu
 * @since 1.0.0
 */
public class SimpleVectorStore extends AbstractObservationVectorStore {

	private static final Log logger = LogFactory.getLog(SimpleVectorStore.class);

	private final JsonMapper jsonMapper;

	private final SimpleVectorStoreFilterExpressionEvaluator filterExpressionEvaluator;

	protected Map<String, SimpleVectorStoreContent> store = new ConcurrentHashMap<>();

	protected SimpleVectorStore(SimpleVectorStoreBuilder builder) {
		super(builder);
		this.jsonMapper = JsonMapper.builder().addModules(JacksonUtils.instantiateAvailableModules()).build();
		this.filterExpressionEvaluator = new SimpleVectorStoreFilterExpressionEvaluator();
	}

	/**
	 * Creates an instance of SimpleVectorStore builder.
	 * @return the SimpleVectorStore builder.
	 */
	public static SimpleVectorStoreBuilder builder(EmbeddingModel embeddingModel) {
		return new SimpleVectorStoreBuilder(embeddingModel);
	}

	@Override
	public void doAdd(List<Document> documents) {
		// 经公开 add 进入时，父类已拒绝媒体文档；这里继续检查列表是否存在且非空。
		Objects.requireNonNull(documents, "Documents list cannot be null");
		if (documents.isEmpty()) {
			throw new IllegalArgumentException("Documents list cannot be empty");
		}

		for (Document document : documents) {
			if (logger.isInfoEnabled()) {
				logger.info("Calling EmbeddingModel for document id = " + document.getId());
			}
			// 当前实现逐文档调用模型，没有使用父类保存的 batchingStrategy 或批量 embed 重载。
			// 文档元数据是否参与向量化由具体 EmbeddingModel 的文档重载决定。
			float[] embedding = this.embeddingModel.embed(document);
			SimpleVectorStoreContent storeContent = new SimpleVectorStoreContent(document.getId(),
					Objects.requireNonNullElse(document.getText(), ""), document.getMetadata(), embedding);
			// 同 id 直接覆盖；循环中某个文档失败时，前面已写入的记录不会回滚。
			this.store.put(document.getId(), storeContent);
		}
	}

	@Override
	public void doDelete(List<String> idList) {
		for (String id : idList) {
			this.store.remove(id);
		}
	}

	@Override
	public void doDelete(Filter.Expression filterExpression) {
		// 按 metadata 选出待删 id 后调用内部删除方法；不计算 embedding，也不按相似度筛选。
		List<String> idList = this.store.values()
			.stream()
			.filter(document -> doFilterPredicate(filterExpression).test(document))
			.map(SimpleVectorStoreContent::getId)
			.toList();
		this.doDelete(idList);
	}

	@Override
	public List<Document> doSimilaritySearch(SearchRequest request) {
		// 先将查询的内容转换成向量
		// 每次查询只生成一个查询向量，文档向量在入库时生成并保存在 store 中。
		float[] userQueryEmbedding = getUserQueryEmbedding(request.getQuery());
		return this.store.values()
			.stream()
			// 1. metadata 布尔条件先筛选候选；此 Predicate 与后面的相似度计算是两种不同职责。
			.filter(document -> doFilterPredicate(request.getFilterExpression()).test(document))
			// 2. 对每个候选算余弦，并重建带 score 的 Document；toDocument 还写入 distance=1-score。
			.map(content -> content
				.toDocument(EmbeddingMath.cosineSimilarity(userQueryEmbedding, content.getEmbedding())))
			// 3. 保留达到阈值的结果。这里没有把余弦从 [-1,1] 映射到 [0,1]，0 阈值仍会排除负分。
			.filter(document -> document.getScore() != null && document.getScore() >= request.getSimilarityThreshold())
			// 4. 按分数降序，再截取最多 topK 条；候选不足时不会补齐，同分记录没有额外稳定排序键。
			.sorted(Comparator.comparing(Document::getScore).reversed())
			.limit(request.getTopK())
			.toList();
	}

	private Predicate<SimpleVectorStoreContent> doFilterPredicate(Filter.@Nullable Expression filterExpression) {
		// 空表达式放行所有记录；非空表达式只检查 metadata，既不读取正文，也不使用向量。
		if (filterExpression == null) {
			return document -> true;
		}
		return document -> this.filterExpressionEvaluator.evaluate(filterExpression, document.getMetadata());
	}

	/**
	 * Serialize the vector store content into a file in JSON format.
	 * @param file the file to save the vector store content
	 */
	public void save(File file) {
		// 快照包含正文、metadata 和已计算的 embedding；此操作不会再次调用模型。
		String json = getVectorDbAsJson();
		try {
			if (!file.exists()) {
				if (logger.isInfoEnabled()) {
					logger.info("Creating new vector store file: " + file);
				}
				try {
					Files.createFile(file.toPath());
				}
				catch (FileAlreadyExistsException e) {
					throw new RuntimeException("File already exists: " + file, e);
				}
				catch (IOException e) {
					throw new RuntimeException("Failed to create new file: " + file + ". Reason: " + e.getMessage(), e);
				}
			}
			else if (logger.isInfoEnabled()) {
				logger.info("Overwriting existing vector store file: " + file);
			}
			try (OutputStream stream = new FileOutputStream(file);
					Writer writer = new OutputStreamWriter(stream, StandardCharsets.UTF_8)) {
				writer.write(json);
				writer.flush();
			}
		}
		catch (IOException ex) {
			logger.error("IOException occurred while saving vector store file.", ex);
			throw new RuntimeException(ex);
		}
		catch (SecurityException ex) {
			logger.error("SecurityException occurred while saving vector store file.", ex);
			throw new RuntimeException(ex);
		}
		catch (NullPointerException ex) {
			logger.error("NullPointerException occurred while saving vector store file.", ex);
			throw new RuntimeException(ex);
		}
	}

	/**
	 * Deserialize the vector store content from a file in JSON format into memory.
	 * @param file the file to load the vector store content
	 */
	public void load(File file) {
		// 读取后整体替换 store，不与已有记录合并，也不重新生成向量。
		// 目标类型为 HashMap，因此加载后不能继续假定 store 具有初始 ConcurrentHashMap 的并发语义。
		TypeReference<HashMap<String, SimpleVectorStoreContent>> typeRef = new TypeReference<>() {

		};
		this.store = this.jsonMapper.readValue(file, typeRef);
	}

	/**
	 * Deserialize the vector store content from a resource in JSON format into memory.
	 * @param resource the resource to load the vector store content
	 */
	public void load(Resource resource) {
		// Resource 与 File 重载采用相同的 HashMap 反序列化目标，都会覆盖现有内存数据。
		TypeReference<HashMap<String, SimpleVectorStoreContent>> typeRef = new TypeReference<>() {

		};
		try {
			this.store = this.jsonMapper.readValue(resource.getInputStream(), typeRef);
		}
		catch (IOException ex) {
			throw new RuntimeException(ex);
		}
	}

	private String getVectorDbAsJson() {
		ObjectWriter objectWriter = this.jsonMapper.writerWithDefaultPrettyPrinter();
		try {
			return objectWriter.writeValueAsString(this.store);
		}
		catch (JacksonException ex) {
			throw new RuntimeException("Error serializing documentMap to JSON.", ex);
		}
	}

	private float[] getUserQueryEmbedding(String query) {
		return this.embeddingModel.embed(query);
	}

	@Override
	public VectorStoreObservationContext.Builder createObservationContextBuilder(String operationName) {

		// 为父类的每次公开操作提供数据库类型、维度、集合和度量等观测属性。
		// dimensions() 默认可能发起模型调用；这一步发生在父类创建 Observation 之前。
		return VectorStoreObservationContext.builder(VectorStoreProvider.SIMPLE.value(), operationName)
			.dimensions(this.embeddingModel.dimensions())
			.collectionName("in-memory-map")
			.similarityMetric(VectorStoreSimilarityMetric.COSINE.value());
	}

	public static final class EmbeddingMath {

		private EmbeddingMath() {
			throw new UnsupportedOperationException("This is a utility class and cannot be instantiated");
		}

		public static double cosineSimilarity(float[] vectorX, float[] vectorY) {
			// 维度相同和非零模长是余弦计算的数学前提，不能替代“来自兼容模型”的语义前提。
			if (vectorX == null || vectorY == null) {
				throw new RuntimeException("Vectors must not be null");
			}
			if (vectorX.length != vectorY.length) {
				throw new IllegalArgumentException("Vectors lengths must be equal");
			}

			float dotProduct = dotProduct(vectorX, vectorY);
			// 此处 norm 返回平方和，真正的模长在最终分母里取平方根。
			float normX = norm(vectorX);
			float normY = norm(vectorY);

			if (normX == 0 || normY == 0) {
				throw new IllegalArgumentException("Vectors cannot have zero norm");
			}

			// 方向越接近，结果通常越大；实现返回原始余弦，没有概率解释或额外区间归一化。
			return dotProduct / (Math.sqrt(normX) * Math.sqrt(normY));
		}

		public static float dotProduct(float[] vectorX, float[] vectorY) {
			if (vectorX.length != vectorY.length) {
				throw new IllegalArgumentException("Vectors lengths must be equal");
			}

			float result = 0;
			for (int i = 0; i < vectorX.length; ++i) {
				result += vectorX[i] * vectorY[i];
			}

			return result;
		}

		public static float norm(float[] vector) {
			return dotProduct(vector, vector);
		}

	}

	public static final class SimpleVectorStoreBuilder extends AbstractVectorStoreBuilder<SimpleVectorStoreBuilder> {

		private SimpleVectorStoreBuilder(EmbeddingModel embeddingModel) {
			super(embeddingModel);
		}

		@Override
		public SimpleVectorStore build() {
			return new SimpleVectorStore(this);
		}

	}

}
