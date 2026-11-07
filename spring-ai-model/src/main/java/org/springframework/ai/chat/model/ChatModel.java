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

import java.util.Arrays;

import org.jspecify.annotations.Nullable;
import reactor.core.publisher.Flux;

import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.Model;

/**
 * 中文导读：所有聊天模型适配器的统一调用协议。
 * 输入 {@link Prompt} 包含消息与选项，输出 {@link ChatResponse} 包含生成结果及元数据。
 * OpenAI 等具体实现负责把这些通用对象映射为供应商请求，再映射回通用响应。
 * <p>
 * 直接调用本接口不会经过 ChatClient 的 Advisor 链，因此不会自动获得其会话记忆、
 * RAG 增强或工具执行循环。模型可以返回“调用某工具”的意图，但执行工具并再次请求模型
 * 是上层 ToolCallingAdvisor/ToolCallingManager 的职责。
 * <p>
 * 两个返回字符串的 call 重载只是便捷入口，只提取第一条 Generation 的文本。
 * 需要用量、结束原因、多候选结果或工具调用信息时，应使用 call(Prompt)。
 */
public interface ChatModel extends Model<Prompt, ChatResponse>, StreamingChatModel {

	default @Nullable String call(String message) {
		Prompt prompt = new Prompt(new UserMessage(message));
		Generation generation = call(prompt).getResult();
		return (generation != null) ? generation.getOutput().getText() : "";
	}

	default @Nullable String call(Message... messages) {
		Prompt prompt = new Prompt(Arrays.asList(messages));
		Generation generation = call(prompt).getResult();
		return (generation != null) ? generation.getOutput().getText() : "";
	}

	@Override
	ChatResponse call(Prompt prompt);

	/**
	 * Gets the chat options for this model.
	 * @return the chat options
	 * @since 2.0.0
	 */
	default ChatOptions getOptions() {
		return ChatOptions.builder().build();
	}

	/**
	 * @deprecated use {@link #getOptions()} instead.
	 */
	@Deprecated(forRemoval = true)
	default ChatOptions getDefaultOptions() {
		return getOptions();
	}

	default Flux<ChatResponse> stream(Prompt prompt) {
		// 同时实现 StreamingChatModel 不代表具体供应商已实现流式；默认行为明确抛出不支持异常。
		throw new UnsupportedOperationException("streaming is not supported");
	}

}
