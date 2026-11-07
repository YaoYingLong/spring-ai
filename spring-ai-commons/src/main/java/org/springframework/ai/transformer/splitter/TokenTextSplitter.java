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

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.EncodingType;
import com.knuddels.jtokkit.api.IntArrayList;

import org.springframework.util.Assert;

/**
 * A {@link TextSplitter} that splits text into chunks of a target size in tokens.
 *
 * <p>
 * 中文导读：先用 JTokkit 编码为 token，再按目标 token 数取候选片段，必要时回退到片段内的标点。
 * 父类负责生成 Document 和继承元数据，本类只负责文本边界；入口为 splitText，核心循环在 doSplit。
 * 默认编码为 CL100K_BASE，Builder 可替换编码类型；它不会根据下游 EmbeddingModel 自动选择 tokenizer。
 *
 * <p>
 * 默认目标为 800 token，标点位置阈值为 350 个 Java 字符，保留文本的长度必须大于 5。
 * 默认识别英文句点、问号、感叹号和换行，Builder 可显式提供其他标点。
 * 算法没有片段重叠；目标 token 数、标点阈值和循环次数上限都不能理解为最终片段的严格大小保证。
 *
 * @author Raphael Yu
 * @author Christian Tzolov
 * @author Ricken Bazolo
 * @author Jemin Huh
 */
public class TokenTextSplitter extends TextSplitter {

	private static final int DEFAULT_CHUNK_SIZE = 800;

	private static final int MIN_CHUNK_SIZE_CHARS = 350;

	private static final int MIN_CHUNK_LENGTH_TO_EMBED = 5;

	private static final int MAX_NUM_CHUNKS = 10000;

	private static final boolean KEEP_SEPARATOR = true;

	private static final List<Character> DEFAULT_PUNCTUATION_MARKS = List.of('.', '?', '!', '\n');

	private static final EncodingType DEFAULT_ENCODING_TYPE = EncodingType.CL100K_BASE;

	private final EncodingRegistry registry = Encodings.newLazyEncodingRegistry();

	private final Encoding encoding;

	// The target size of each text chunk in tokens
	private final int chunkSize;

	// The minimum size of each text chunk in characters
	// 实际用于判断标点下标是否足够靠后，不会补齐短尾块；单位是 String 的 UTF-16 下标。
	private final int minChunkSizeChars;

	// Discard chunks shorter than this
	// 代码使用严格大于比较，因此长度恰好等于该值的片段也会丢弃。
	private final int minChunkLengthToEmbed;

	// The maximum number of chunks to generate from a text
	// 实际限制主循环计数；循环结束后还可能额外添加一个包含全部剩余 token 的尾块。
	private final int maxNumChunks;

	private final boolean keepSeparator;

	private final List<Character> punctuationMarks;

	/**
	 * @deprecated since 2.0.0-M3, use {@link #builder()} instead.
	 */
	@Deprecated(since = "2.0.0-M3", forRemoval = true)
	@SuppressWarnings("deprecation")
	public TokenTextSplitter() {
		this(DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE_CHARS, MIN_CHUNK_LENGTH_TO_EMBED, MAX_NUM_CHUNKS, KEEP_SEPARATOR,
				DEFAULT_PUNCTUATION_MARKS);
	}

	/**
	 * @deprecated since 2.0.0-M3, use {@link #builder()} instead.
	 */
	@Deprecated(since = "2.0.0-M3", forRemoval = true)
	public TokenTextSplitter(boolean keepSeparator) {
		this(DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE_CHARS, MIN_CHUNK_LENGTH_TO_EMBED, MAX_NUM_CHUNKS, keepSeparator,
				DEFAULT_PUNCTUATION_MARKS);
	}

	/**
	 * @deprecated since 2.0.0-M3, use {@link #builder()} instead.
	 */
	@Deprecated(since = "2.0.0-M3", forRemoval = true)
	public TokenTextSplitter(EncodingType encodingType) {
		this(encodingType, DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE_CHARS, MIN_CHUNK_LENGTH_TO_EMBED, MAX_NUM_CHUNKS,
				KEEP_SEPARATOR, DEFAULT_PUNCTUATION_MARKS);
	}

	/**
	 * @deprecated since 2.0.0-M3, use {@link #builder()} instead.
	 */
	@Deprecated(since = "2.0.0-M3", forRemoval = true)
	public TokenTextSplitter(EncodingType encodingType, boolean keepSeparator) {
		this(encodingType, DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE_CHARS, MIN_CHUNK_LENGTH_TO_EMBED, MAX_NUM_CHUNKS,
				keepSeparator, DEFAULT_PUNCTUATION_MARKS);
	}

	/**
	 * @deprecated since 2.0.0-M3, use {@link #builder()} instead.
	 */
	@Deprecated(since = "2.0.0-M3", forRemoval = true)
	public TokenTextSplitter(int chunkSize, int minChunkSizeChars, int minChunkLengthToEmbed, int maxNumChunks,
			boolean keepSeparator, List<Character> punctuationMarks) {
		this(DEFAULT_ENCODING_TYPE, chunkSize, minChunkSizeChars, minChunkLengthToEmbed, maxNumChunks, keepSeparator,
				punctuationMarks);
	}

	private TokenTextSplitter(EncodingType encodingType, int chunkSize, int minChunkSizeChars,
			int minChunkLengthToEmbed, int maxNumChunks, boolean keepSeparator, List<Character> punctuationMarks) {
		Assert.notNull(encodingType, "encodingType must not be null");
		Assert.isTrue(chunkSize > 0, "chunkSize must be greater than zero");
		Assert.isTrue(maxNumChunks > 0, "maxNumChunks must be greater than zero");
		Assert.isTrue(minChunkSizeChars >= 0, "minChunkSizeChars must not be negative");
		Assert.isTrue(minChunkLengthToEmbed >= 0, "minChunkLengthToEmbed must not be negative");
		// Registry 延迟创建编码，但构造当前 splitter 时已在这里请求选定编码，不是首次 split 才请求。
		this.encoding = this.registry.getEncoding(encodingType);
		this.chunkSize = chunkSize;
		this.minChunkSizeChars = minChunkSizeChars;
		this.minChunkLengthToEmbed = minChunkLengthToEmbed;
		this.maxNumChunks = maxNumChunks;
		this.keepSeparator = keepSeparator;
		Assert.notEmpty(punctuationMarks, "punctuationMarks must not be empty");
		this.punctuationMarks = punctuationMarks;
	}

	public static Builder builder() {
		return new Builder();
	}

	@Override
	protected List<String> splitText(String text) {
		return doSplit(text, this.chunkSize);
	}

	/**
	 * Splits text into chunks based on token count.
	 * <p>
	 * Punctuation-based splitting only applies when the token count exceeds the chunk
	 * size ({@code tokens.size() > chunkSize}). Text that exactly matches or is smaller
	 * than the chunk size is returned as a single chunk without punctuation-based
	 * truncation.
	 * @param text the text to split
	 * @param chunkSize the target chunk size in tokens
	 * @return list of text chunks
	 */
	protected List<String> doSplit(String text, int chunkSize) {
		// 空白正文不产出 Document；这也决定了父类传入的空文本如何被过滤。
		if (text.trim().isEmpty()) {
			return new ArrayList<>();
		}

		// 一次编码整段输入，后续 tokens 表示尚未消费的后缀；不是按字符数直接截取原字符串。
		List<Integer> tokens = getEncodedTokens(text);
		List<String> chunks = new ArrayList<>();
		int num_chunks = 0;
		while (!tokens.isEmpty() && num_chunks < this.maxNumChunks) {
			// 先取最多 chunkSize 个 token，再解码为可查找标点的文本。
			List<Integer> chunk = tokens.subList(0, Math.min(chunkSize, tokens.size()));
			String chunkText = decodeTokens(chunk);

			// Skip the chunk if it is empty or whitespace
			if (chunkText.trim().isEmpty()) {
				// 空白候选仍然消费 token，但 continue 会跳过下方的 num_chunks 计数。
				tokens = tokens.subList(chunk.size(), tokens.size());
				continue;
			}

			// Only apply punctuation-based truncation if we have more tokens than the
			// chunk size
			// This prevents unnecessary splitting of small texts
			if (tokens.size() > chunkSize) {
				// 只有剩余输入超过目标大小才寻找断点；恰好一块或更短的尾块不会因标点再被拆分。
				// Find the last period or punctuation mark in the chunk
				int lastPunctuation = getLastPunctuationIndex(chunkText);

				if (lastPunctuation != -1 && lastPunctuation > this.minChunkSizeChars) {
					// 使用最后一个已配置标点，且下标必须严格大于阈值；substring 保留该标点。
					// Truncate the chunk text at the punctuation mark
					chunkText = chunkText.substring(0, lastPunctuation + 1);
				}
			}

			// 两种模式都会 trim；keepSeparator=false 仅把平台换行符替换为空格，不会删除所有标点。
			String chunkTextToAppend = (this.keepSeparator) ? chunkText.trim()
					: chunkText.replace(System.lineSeparator(), " ").trim();
			if (chunkTextToAppend.length() > this.minChunkLengthToEmbed) {
				chunks.add(chunkTextToAppend);
			}

			// 按截断后、清理空白前的文本重新编码来计算消费量，不把已消费内容重叠到下一片段。
			// 即使片段因过短而没有加入 chunks，也照常前移，因此短片段不会合并回后一个片段。
			// Remove the tokens corresponding to the chunk text from the remaining tokens
			tokens = tokens.subList(getEncodedTokens(chunkText).size(), tokens.size());

			num_chunks++;
		}

		// Handle the remaining tokens
		if (!tokens.isEmpty()) {
			// 达到主循环上限后，剩余内容一次性输出，可能远大于 chunkSize，且总块数可能多一块。
			// 这条尾块路径总会替换平台换行符，不读取 keepSeparator。
			String remaining_text = decodeTokens(tokens).replace(System.lineSeparator(), " ").trim();
			if (remaining_text.length() > this.minChunkLengthToEmbed) {
				chunks.add(remaining_text);
			}
		}

		return chunks;
	}

	protected int getLastPunctuationIndex(String chunkText) {
		// 分别找每种标点的最后位置再取最大值；没有句子语义分析，也不会自动识别未配置的中文标点。
		// find the max index of any punctuation mark
		int maxLastPunctuation = -1;
		for (Character punctuationMark : this.punctuationMarks) {
			int lastPunctuation = chunkText.lastIndexOf(punctuationMark);
			maxLastPunctuation = Math.max(maxLastPunctuation, lastPunctuation);
		}
		return maxLastPunctuation;
	}

	private List<Integer> getEncodedTokens(String text) {
		Assert.notNull(text, "Text must not be null");
		return this.encoding.encode(text).boxed();
	}

	private String decodeTokens(List<Integer> tokens) {
		Assert.notNull(tokens, "Tokens must not be null");
		var tokensIntArray = new IntArrayList(tokens.size());
		tokens.forEach(tokensIntArray::add);
		return this.encoding.decode(tokensIntArray);
	}

	public static final class Builder {

		private EncodingType encodingType = DEFAULT_ENCODING_TYPE;

		private int chunkSize = DEFAULT_CHUNK_SIZE;

		private int minChunkSizeChars = MIN_CHUNK_SIZE_CHARS;

		private int minChunkLengthToEmbed = MIN_CHUNK_LENGTH_TO_EMBED;

		private int maxNumChunks = MAX_NUM_CHUNKS;

		private boolean keepSeparator = KEEP_SEPARATOR;

		private List<Character> punctuationMarks = DEFAULT_PUNCTUATION_MARKS;

		private Builder() {
		}

		public Builder withEncodingType(EncodingType encodingType) {
			this.encodingType = encodingType;
			return this;
		}

		public Builder withChunkSize(int chunkSize) {
			this.chunkSize = chunkSize;
			return this;
		}

		public Builder withMinChunkSizeChars(int minChunkSizeChars) {
			this.minChunkSizeChars = minChunkSizeChars;
			return this;
		}

		public Builder withMinChunkLengthToEmbed(int minChunkLengthToEmbed) {
			this.minChunkLengthToEmbed = minChunkLengthToEmbed;
			return this;
		}

		public Builder withMaxNumChunks(int maxNumChunks) {
			this.maxNumChunks = maxNumChunks;
			return this;
		}

		public Builder withKeepSeparator(boolean keepSeparator) {
			this.keepSeparator = keepSeparator;
			return this;
		}

		public Builder withPunctuationMarks(List<Character> punctuationMarks) {
			this.punctuationMarks = punctuationMarks;
			return this;
		}

		public TokenTextSplitter build() {
			// 参数合法性由构造器统一检查，编码和各边界策略在构建时确定。
			return new TokenTextSplitter(this.encodingType, this.chunkSize, this.minChunkSizeChars,
					this.minChunkLengthToEmbed, this.maxNumChunks, this.keepSeparator, this.punctuationMarks);
		}

	}

}
