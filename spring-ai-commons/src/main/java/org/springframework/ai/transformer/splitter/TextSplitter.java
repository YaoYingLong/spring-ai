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

package org.springframework.ai.transformer.splitter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jspecify.annotations.Nullable;

import org.springframework.ai.document.ContentFormatter;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.DocumentTransformer;

/**
 * 中文导读：ETL 的文本分块模板，负责把 Document 列表转换为片段 Document 列表。
 * 典型调用链是 Reader 产出文档、split/apply 切块、VectorStore.accept/add 写入。
 * 本类负责文档属性的传递，具体如何确定文本边界交给 {@link #splitText(String)}。
 *
 * <p>
 * 阅读顺序为 apply、doSplitDocuments、createDocuments，最后进入具体 splitter 的 splitText。
 * 每个输出片段有新 id，同时通过 parent_document_id、chunk_index、total_chunks 关联原文。
 * 本层既不计算 embedding，也不根据模型上下文窗口自动调整片段大小。
 */
public abstract class TextSplitter implements DocumentTransformer {

	private static final Log logger = LogFactory.getLog(TextSplitter.class);

	/**
	 * If true the children documents inherit the content formatter of the parent they
	 * were split from.
	 */
	private boolean copyContentFormatter = true;

	@Override
	public List<Document> apply(List<Document> documents) {
		// DocumentTransformer 的函数式入口；两个 split 重载最终都走同一条转换路径。
		return doSplitDocuments(documents);
	}

	public List<Document> split(List<Document> documents) {
		return this.apply(documents);
	}

	public List<Document> split(Document document) {
		return this.apply(List.of(document));
	}

	public boolean isCopyContentFormatter() {
		return this.copyContentFormatter;
	}

	public void setCopyContentFormatter(boolean copyContentFormatter) {
		this.copyContentFormatter = copyContentFormatter;
	}

	private List<Document> doSplitDocuments(List<Document> documents) {
		// 按输入次序分别收集正文和附属属性，各列表的相同下标始终对应同一个原文档。
		List<String> texts = new ArrayList<>();
		List<Map<String, Object>> metadataList = new ArrayList<>();
		List<ContentFormatter> formatters = new ArrayList<>();
		List<@Nullable Double> scores = new ArrayList<>();
		List<String> originalIds = new ArrayList<>();

		for (Document doc : documents) {
			// 这里只读取 text；媒体文档退化为空文本，不执行 OCR、媒体解析或多模态分块。
			// 对 TokenTextSplitter 而言，空文本不会产生任何输出片段。
			texts.add(Objects.requireNonNullElse(doc.getText(), ""));
			metadataList.add(doc.getMetadata());
			formatters.add(doc.getContentFormatter());
			scores.add(doc.getScore());
			originalIds.add(doc.getId());
		}

		return createDocuments(texts, formatters, metadataList, scores, originalIds);
	}

	private List<Document> createDocuments(List<String> texts, List<ContentFormatter> formatters,
			List<Map<String, Object>> metadataList, List<@Nullable Double> scores, List<String> originalIds) {

		// Process the data in a column oriented way and recreate the Document
		List<Document> documents = new ArrayList<>();

		for (int i = 0; i < texts.size(); i++) {
			String text = texts.get(i);
			Map<String, Object> metadata = metadataList.get(i);
			Double originalScore = scores.get(i);
			String originalId = originalIds.get(i);

			// 扩展点只返回字符串；下面统一重建 Document，使所有 splitter 共享元数据传递规则。
			List<String> chunks = splitText(text);
			if (chunks.size() > 1) {
				if (logger.isInfoEnabled()) {
					logger.info("Splitting up document into " + chunks.size() + " chunks.");
				}
			}

			for (int chunkIndex = 0; chunkIndex < chunks.size(); chunkIndex++) {
				String chunk = chunks.get(chunkIndex);

				// 为每个片段创建独立 Map；元数据的值仍是浅复制，后面三个分块字段会覆盖同名旧值。
				Map<String, Object> enhancedMetadata = metadata.entrySet()
					.stream()
					// filter left here despite explicit JSpecify disallowing nulls for
					// now.
					.filter(e -> e.getKey() != null && e.getValue() != null)
					.collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

				enhancedMetadata.put("parent_document_id", originalId);
				enhancedMetadata.put("chunk_index", chunkIndex);
				enhancedMetadata.put("total_chunks", chunks.size());

				// 不复用父 id，避免多个片段在按 id 存储时互相覆盖；chunk_index 从 0 开始。
				// 原 score 仅随数据传递，没有针对片段重新计算相关性。
				Document newDoc = Document.builder()
					.text(chunk)
					.metadata(enhancedMetadata)
					.score(originalScore)
					.build();

				if (this.copyContentFormatter) {
					// 传递同一个 formatter 引用，保证父文档的内容呈现策略可延续到片段。
					// Transfer the content-formatter of the parent to the chunked
					// documents it was split into.
					newDoc.setContentFormatter(formatters.get(i));
				}

				documents.add(newDoc);
			}
		}
		return documents;
	}

	/**
	 * 根据具体算法返回按顺序排列的片段文本；返回空列表表示该原文档不进入后续写入阶段。
	 */
	protected abstract List<String> splitText(String text);

}
