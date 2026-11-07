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

package org.springframework.ai.chat.client.advisor;

import java.util.Map;

import org.jspecify.annotations.Nullable;

import org.springframework.ai.chat.client.ChatClientAttributes;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.CallAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.StructuredOutputChatOptions;
import org.springframework.core.Ordered;
import org.springframework.util.Assert;
import org.springframework.util.StringUtils;

/**
 * A {@link CallAdvisor} that uses a {@link ChatModel} to generate a response.
 * <p>
 * 中文导读：同步 Advisor 链的终点。上游已完成记忆、检索、工具上下文等处理，
 * 这里将最终 Prompt 交给模型，并把请求 context 带回响应供上游读取。
 * 本节点不再调用 nextCall，getOrder() 使用最低优先级，使它处于常规 Advisor 之后。
 * 工具循环的多轮请求会多次进入这个终点，但每次只执行一次 ChatModel.call。
 *
 * @author Thomas Vitale
 * @author Christian Tzolov
 * @since 1.0.0
 */
public final class ChatModelCallAdvisor implements CallAdvisor {

	private final ChatModel chatModel;

	private ChatModelCallAdvisor(ChatModel chatModel) {
		Assert.notNull(chatModel, "chatModel cannot be null");
		this.chatModel = chatModel;
	}

	@Override
	public ChatClientResponse adviseCall(ChatClientRequest chatClientRequest, CallAdvisorChain callAdvisorChain) {
		Assert.notNull(chatClientRequest, "the chatClientRequest cannot be null");

		ChatClientRequest formattedChatClientRequest = augmentWithFormatInstructions(chatClientRequest);

		ChatResponse chatResponse = this.chatModel.call(formattedChatClientRequest.prompt());

		return ChatClientResponse.builder()
			.chatResponse(chatResponse)
			.context(Map.copyOf(formattedChatClientRequest.context()))
			.build();
	}

	private static ChatClientRequest augmentWithFormatInstructions(ChatClientRequest chatClientRequest) {

		String outputFormat = (String) chatClientRequest.context().get(ChatClientAttributes.OUTPUT_FORMAT.getKey());

		String outputSchema = (String) chatClientRequest.context()
			.get(ChatClientAttributes.STRUCTURED_OUTPUT_SCHEMA.getKey());

		if (!StringUtils.hasText(outputFormat) && !StringUtils.hasText(outputSchema)) {
			return chatClientRequest;
		}

		// entity(...) 先将格式/Schema 放入 context。原生模式且选项支持 Schema 时，通过模型参数传递。
		// 其余情况将格式说明追加到最后一条用户消息，依靠提示词约束模型输出。
		boolean usesNativeStructuredOutput = chatClientRequest.context()
			.containsKey(ChatClientAttributes.STRUCTURED_OUTPUT_NATIVE.getKey());

		if (usesNativeStructuredOutput && StringUtils.hasText(outputSchema) && chatClientRequest.prompt()
			.getOptions() instanceof StructuredOutputChatOptions structuredOutputChatOptions) {

			// 从原有选项派生新对象，保留模型及工具配置；这里不执行 Schema 校验或反序列化。
			var augmentedOptions = structuredOutputChatOptions.mutate().outputSchema(outputSchema).build();
			Prompt augmentedPrompt = chatClientRequest.prompt().mutate().chatOptions(augmentedOptions).build();

			return ChatClientRequest.builder()
				.prompt(augmentedPrompt)
				.context(Map.copyOf(chatClientRequest.context()))
				.build();
		}

		Prompt augmentedPrompt = chatClientRequest.prompt()
			.augmentUserMessage(userMessage -> userMessage.mutate()
				.text(userMessage.getText() + System.lineSeparator() + outputFormat)
				.build());

		return ChatClientRequest.builder()
			.prompt(augmentedPrompt)
			.context(Map.copyOf(chatClientRequest.context()))
			.build();
	}

	@Override
	public String getName() {
		return "call";
	}

	@Override
	public int getOrder() {
		return Ordered.LOWEST_PRECEDENCE;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private @Nullable ChatModel chatModel;

		private Builder() {
		}

		public Builder chatModel(ChatModel chatModel) {
			this.chatModel = chatModel;
			return this;
		}

		public ChatModelCallAdvisor build() {
			Assert.state(this.chatModel != null, "chatModel cannot be null");
			return new ChatModelCallAdvisor(this.chatModel);
		}

	}

}
