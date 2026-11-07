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

package org.springframework.ai.chat.prompt;

import java.util.List;

import org.jspecify.annotations.Nullable;

import org.springframework.ai.model.ModelOptions;

/**
 * {@link ModelOptions} representing the common options that are portable across different
 * chat models.
 * <p>
 * 中文导读：这里定义不同供应商共享的生成参数，具体支持范围仍由模型决定。
 * null 通常表示本层未指定；默认值如何补齐取决于调用入口和具体适配器。
 * ChatClient 会先复制 ChatModel.getOptions()，再合并请求 Builder 的非 null 配置。
 * 直接调用 ChatModel 时，应阅读该实现的选项处理，不能假设存在同样的合并过程。
 * <p>
 * mutate() 必须保留具体 Builder 类型，才能在通用调用链中继续传递供应商字段、
 * ToolCallingChatOptions 的工具配置及 StructuredOutputChatOptions 的输出 Schema。
 */
public interface ChatOptions extends ModelOptions {

	/**
	 * Returns the model to use for the chat.
	 * @return the model to use for the chat
	 */
	@Nullable String getModel();

	/**
	 * Returns the frequency penalty to use for the chat.
	 * @return the frequency penalty to use for the chat
	 */
	@Nullable Double getFrequencyPenalty();

	/**
	 * Returns the maximum number of tokens to use for the chat.
	 * @return the maximum number of tokens to use for the chat
	 */
	@Nullable Integer getMaxTokens();

	/**
	 * Returns the presence penalty to use for the chat.
	 * @return the presence penalty to use for the chat
	 */
	@Nullable Double getPresencePenalty();

	/**
	 * Returns the stop sequences to use for the chat.
	 * @return the stop sequences to use for the chat
	 */
	@Nullable List<String> getStopSequences();

	/**
	 * Returns the temperature to use for the chat.
	 * @return the temperature to use for the chat
	 */
	@Nullable Double getTemperature();

	/**
	 * Returns the top K to use for the chat.
	 * @return the top K to use for the chat
	 */
	@Nullable Integer getTopK();

	/**
	 * Returns the top P to use for the chat.
	 * @return the top P to use for the chat
	 */
	@Nullable Double getTopP();

	/**
	 * Returns a new {@link Builder} initialized with the values of this
	 * {@link ChatOptions}.
	 *
	 * Concrete ChatOptions classes must implement this and return the most concrete
	 * builder implementation.
	 */
	ChatOptions.Builder<?> mutate();

	/**
	 * Creates a new {@link Builder} to create the default {@link ChatOptions}.
	 * @return Returns a new {@link Builder}.
	 */
	static ChatOptions.Builder<?> builder() {
		return new DefaultChatOptionsBuilder<>();
	}

	/**
	 * Builder for creating {@link ChatOptions} instance.
	 */
	interface Builder<B extends Builder<B>> extends Cloneable {

		B clone();

		/**
		 * Builds with the model to use for the chat.
		 * @param model
		 * @return the builder
		 */
		B model(@Nullable String model);

		/**
		 * Builds with the frequency penalty to use for the chat.
		 * @param frequencyPenalty
		 * @return the builder.
		 */
		B frequencyPenalty(@Nullable Double frequencyPenalty);

		/**
		 * Builds with the maximum number of tokens to use for the chat.
		 * @param maxTokens
		 * @return the builder.
		 */
		B maxTokens(@Nullable Integer maxTokens);

		/**
		 * Builds with the presence penalty to use for the chat.
		 * @param presencePenalty
		 * @return the builder.
		 */
		B presencePenalty(@Nullable Double presencePenalty);

		/**
		 * Builds with the stop sequences to use for the chat.
		 * @param stopSequences
		 * @return the builder.
		 */
		B stopSequences(@Nullable List<String> stopSequences);

		/**
		 * Builds with the temperature to use for the chat.
		 * @param temperature
		 * @return the builder.
		 */
		B temperature(@Nullable Double temperature);

		/**
		 * Builds with the top K to use for the chat.
		 * @param topK
		 * @return the builder.
		 */
		B topK(@Nullable Integer topK);

		/**
		 * Builds with the top P to use for the chat.
		 * @param topP
		 * @return the builder.
		 */
		B topP(@Nullable Double topP);

		/**
		 * Build the {@link ChatOptions}.
		 * @return the Chat options.
		 */
		ChatOptions build();

		/**
		 * Mutate this builder by taking all {@code other}'s values that are non-null,
		 * retaining {@code this} other values.
		 */
		B combineWith(ChatOptions.Builder<?> other);

	}

}
