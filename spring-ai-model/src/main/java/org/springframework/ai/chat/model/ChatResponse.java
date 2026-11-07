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

package org.springframework.ai.chat.model;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import org.jspecify.annotations.Nullable;

import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.model.ModelResponse;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;

/**
 * The chat completion (e.g. generation) response returned by an AI provider.
 * <p>
 * 中文导读：模型层的统一响应容器。generations 保存候选输出，每个 Generation 包含
 * AssistantMessage 及结束原因等生成元数据；chatResponseMetadata 保存请求 ID、模型名、
 * token 用量等响应级信息。它不包含 ChatClient 的 Advisor context。
 * <p>
 * getResult() 只是取第一个候选，而不是合并所有结果。流式情况下每次返回的本类对象
 * 可能仅包含增量文本、部分内容或用量信息，需要结合具体适配器及 MessageAggregator 阅读。
 * 工具调用也是合法输出：AssistantMessage 可携带工具名称、调用 ID 和 JSON 参数，
 * 此时没有普通回答文本并不代表请求失败。
 *
 * @author Christian Tzolov
 * @author Mark Pollack
 * @author Soby Chacko
 * @author John Blum
 * @author Alexandros Pappas
 * @author Thomas Vitale
 */
public class ChatResponse implements ModelResponse<Generation> {

	private final ChatResponseMetadata chatResponseMetadata;

	/**
	 * List of generated messages returned by the AI provider.
	 */
	private final List<Generation> generations;

	/**
	 * Construct a new {@link ChatResponse} instance without metadata.
	 * @param generations the {@link List} of {@link Generation} returned by the AI
	 * provider.
	 */
	public ChatResponse(List<Generation> generations) {
		this(generations, new ChatResponseMetadata());
	}

	/**
	 * Construct a new {@link ChatResponse} instance.
	 * @param generations the {@link List} of {@link Generation} returned by the AI
	 * provider.
	 * @param chatResponseMetadata {@link ChatResponseMetadata} containing information
	 * about the use of the AI provider's API.
	 */
	public ChatResponse(List<Generation> generations, ChatResponseMetadata chatResponseMetadata) {
		Assert.notNull(generations, "'generations' must not be null");
		this.chatResponseMetadata = Objects.requireNonNullElse(chatResponseMetadata, new ChatResponseMetadata());
		this.generations = List.copyOf(generations);
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * The {@link List} of {@link Generation generated outputs}.
	 * <p>
	 * It is a {@link List} of {@link List lists} because the Prompt could request
	 * multiple output {@link Generation generations}.
	 * @return the {@link List} of {@link Generation generated outputs}.
	 */

	@Override
	public List<Generation> getResults() {
		return this.generations;
	}

	/**
	 * @return Returns the first {@link Generation} in the generations list.
	 */
	public @Nullable Generation getResult() {
		if (CollectionUtils.isEmpty(this.generations)) {
			return null;
		}
		return this.generations.get(0);
	}

	/**
	 * @return Returns {@link ChatResponseMetadata} containing information about the use
	 * of the AI provider's API.
	 */
	@Override
	public ChatResponseMetadata getMetadata() {
		return this.chatResponseMetadata;
	}

	/**
	 * Whether the model has requested the execution of a tool.
	 */
	public boolean hasToolCalls() {
		// 检查模型是否提出工具调用意图，不表示工具已经执行成功；执行结果在后续 ToolResponseMessage 中。
		if (CollectionUtils.isEmpty(this.generations)) {
			return false;
		}
		return this.generations.stream().anyMatch(generation -> generation.getOutput().hasToolCalls());
	}

	/**
	 * Whether the model has finished with any of the given finish reasons.
	 */
	public boolean hasFinishReasons(Set<String> finishReasons) {
		Assert.notNull(finishReasons, "finishReasons cannot be null");
		if (CollectionUtils.isEmpty(this.generations)) {
			return false;
		}
		return this.generations.stream().anyMatch(generation -> {
			var finishReason = (generation.getMetadata().getFinishReason() != null)
					? generation.getMetadata().getFinishReason() : "";
			return finishReasons.stream().anyMatch(fr -> fr.equalsIgnoreCase(finishReason));
		});
	}

	@Override
	public String toString() {
		return "ChatResponse [metadata=" + this.chatResponseMetadata + ", generations=" + this.generations + "]";
	}

	@Override
	public boolean equals(@Nullable Object o) {
		if (this == o) {
			return true;
		}
		if (!(o instanceof ChatResponse that)) {
			return false;
		}
		return Objects.equals(this.chatResponseMetadata, that.chatResponseMetadata)
				&& Objects.equals(this.generations, that.generations);
	}

	@Override
	public int hashCode() {
		return Objects.hash(this.chatResponseMetadata, this.generations);
	}

	public static final class Builder {

		private @Nullable List<Generation> generations;

		private ChatResponseMetadata.Builder chatResponseMetadataBuilder;

		private Builder() {
			this.chatResponseMetadataBuilder = ChatResponseMetadata.builder();
		}

		public Builder from(ChatResponse other) {
			this.generations = other.generations;
			return this.metadata(other.chatResponseMetadata);
		}

		public Builder metadata(String key, Object value) {
			this.chatResponseMetadataBuilder.keyValue(key, value);
			return this;
		}

		public Builder metadata(ChatResponseMetadata other) {
			this.chatResponseMetadataBuilder.model(other.getModel());
			this.chatResponseMetadataBuilder.id(other.getId());
			this.chatResponseMetadataBuilder.rateLimit(other.getRateLimit());
			this.chatResponseMetadataBuilder.usage(other.getUsage());
			this.chatResponseMetadataBuilder.promptMetadata(other.getPromptMetadata());
			Set<Map.Entry<String, Object>> entries = other.entrySet();
			for (Map.Entry<String, Object> entry : entries) {
				this.chatResponseMetadataBuilder.keyValue(entry.getKey(), entry.getValue());
			}
			return this;
		}

		public Builder generations(List<Generation> generations) {
			this.generations = generations;
			return this;

		}

		public ChatResponse build() {
			Assert.notNull(this.generations, "'generations' must not be null");
			return new ChatResponse(this.generations, this.chatResponseMetadataBuilder.build());
		}

	}

}
