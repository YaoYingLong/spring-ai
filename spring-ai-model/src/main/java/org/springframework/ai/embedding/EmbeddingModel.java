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

package org.springframework.ai.embedding;

import org.jspecify.annotations.Nullable;
import org.springframework.ai.document.Document;
import org.springframework.ai.model.Model;
import org.springframework.util.Assert;

import java.util.ArrayList;
import java.util.List;

/**
 * EmbeddingModel is a generic interface for embedding models.
 *
 * <p>
 * 中文导读：将文本或文档映射到数值向量的模型接口，是文档入库与查询向量化的共同依赖。
 * 阅读时先看 call 的提供商扩展点，再看字符串、文档和批量重载如何组织 EmbeddingRequest。
 * embedding 用于在同一向量空间中比较相关性，本身不保存原文，也不生成回答或判定事实真假。
 *
 * <p>
 * 文档向量与查询向量需要使用兼容的模型、维度和预处理约定，单纯数组长度相同并不保证语义空间相同。
 * 本接口不记录入库时的模型身份，也不负责验证跨模型或跨版本向量是否可以混用。
 * 文档路径可以把 metadata 纳入输入，字符串查询路径则直接使用调用方提供的文本。
 *
 * @author Mark Pollack
 * @author Christian Tzolov
 * @author Josh Long
 * @author Soby Chacko
 * @author Jihoon Kim
 * @since 1.0.0
 *
 */
public interface EmbeddingModel extends Model<EmbeddingRequest, EmbeddingResponse> {

	@Override
	// 提供商实现负责实际推理、选项合并和响应转换；接口的便利方法最终会调用这个边界。
	EmbeddingResponse call(EmbeddingRequest request);

	/**
	 * Embeds the given text into a vector.
	 * @param text the text to embed.
	 * @return the embedded vector.
	 */
	default float[] embed(String text) {
		Assert.notNull(text, "Text must not be null");
		// 单文本转成只有一个元素的批请求，再取首个输出；没有在此缓存向量或检查响应条数。
		List<float[]> response = this.embed(List.of(text));
		return response.iterator().next();
	}

	/**
	 * Embeds the given document's content into a vector.
	 * <p>
	 * 文档重载由实现类决定内容提取方式；批量文档重载不会逐个调用这个方法，
	 * 而是通过 getEmbeddingContent 提取字符串后统一 call。
	 * @param document the document to embed.
	 * @return the embedded vector.
	 */
	float[] embed(Document document);

	/**
	 * Extracts the text content from a {@link Document} to be used for embedding. By
	 * default, returns {@link Document#getText()}. Implementations that support
	 * {@link org.springframework.ai.document.MetadataMode} should override this method to
	 * return
	 * {@link Document#getFormattedContent(org.springframework.ai.document.MetadataMode)}
	 * with the appropriate metadata mode, so that metadata is included in the text sent
	 * to the embedding API.
	 * @param document the document to extract embedding content from.
	 * @return the text content to embed.
	 */
	default @Nullable String getEmbeddingContent(Document document) {
		Assert.notNull(document, "Document must not be null");
		// 默认只取正文；需要保留来源等语义信息的实现可覆写为按 MetadataMode 格式化的文本。
		// Document 也能携带媒体，因此默认结果允许 null；具体模型负责自身输入能力边界。
		return document.getText();
	}

	/**
	 * Embeds a batch of texts into vectors.
	 * @param texts list of texts to embed.
	 * @return list of embedded vectors.
	 */
	default List<float[]> embed(List<String> texts) {
		Assert.notNull(texts, "Texts must not be null");
		// 这里只发起一次请求，不自动分批；空选项对象的补全规则由具体实现决定。
		// 只保留向量数组，需要 token 用量等响应元数据时应查看 embedForResponse。
		return this.call(new EmbeddingRequest(texts, EmbeddingOptions.builder().build()))
			.getResults()
			.stream()
			.map(Embedding::getOutput)
			.toList();
	}

	/**
	 * Embeds a batch of {@link Document}s into vectors based on a
	 * {@link BatchingStrategy}.
	 * @param documents list of {@link Document}s.
	 * @param options {@link EmbeddingOptions}.
	 * @param batchingStrategy {@link BatchingStrategy}.
	 * @return a list of float[] that represents the vectors for the incoming
	 * {@link Document}s. The returned list is expected to be in the same order of the
	 * {@link Document} list.
	 */
	default List<float[]> embed(List<Document> documents, @Nullable EmbeddingOptions options,
			BatchingStrategy batchingStrategy) {
		Assert.notNull(documents, "Documents must not be null");
		List<float[]> embeddings = new ArrayList<>(documents.size());
		// 可以配置批量策略，多少个文档一个批次转换
		// BatchingStrategy 先划分文档列表；例如 TokenCountBatchingStrategy 按估算 token 总量分批。
		// 分批不等于切块：单个文档超出该策略预算时会报错，文本切分应在上游 ETL 完成。
		List<List<Document>> batch = batchingStrategy.batch(documents);
		for (List<Document> subBatch : batch) {
			// 各批串行执行；提取入口统一走 getEmbeddingContent，保留提供商的 metadata 处理策略。
			// 批策略的 token 估算 formatter 与实际发送内容可能不同，配置时应对齐两者的用途。
			List<String> texts = subBatch.stream().map(this::getEmbeddingContent).toList();
			EmbeddingRequest request = new EmbeddingRequest(texts, options);
			EmbeddingResponse response = this.call(request);
			// 按响应列表下标追加，不依据 Embedding.index 重新排序。
			// 因此批策略必须保持输入次序，提供商返回结果也必须与该批文本次序对应。
			for (int i = 0; i < subBatch.size(); i++) {
				embeddings.add(response.getResults().get(i).getOutput());
			}
		}
		// 校验最终收集数量与原文档数量一致；并不额外验证每个向量的维度或文档对应关系。
		Assert.isTrue(embeddings.size() == documents.size(),
				"Embeddings must have the same number as that of the documents");
		return embeddings;
	}

	/**
	 * Embeds a batch of texts into vectors and returns the {@link EmbeddingResponse}.
	 * @param texts list of texts to embed.
	 * @return the embedding response.
	 */
	default EmbeddingResponse embedForResponse(List<String> texts) {
		Assert.notNull(texts, "Texts must not be null");
		return this.call(new EmbeddingRequest(texts, EmbeddingOptions.builder().build()));
	}

	/**
	 * Get the number of dimensions of the embedded vectors. Note that by default, this
	 * method will call the remote Embedding endpoint to get the dimensions of the
	 * embedded vectors. If the dimensions are known ahead of time, it is recommended to
	 * override this method.
	 * @return the number of dimensions of the embedded vectors.
	 */
	default int dimensions() {
		// 默认通过真实 embedding 结果探测维度，可能产生远程请求和费用；不是纯本地属性读取。
		// 具体实现可覆写并缓存结果，调用者不能仅凭接口名认定此方法无 I/O。
		return embed("Test String").length;
	}

}
