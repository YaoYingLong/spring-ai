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

package org.springframework.ai.chat.memory;

import java.util.List;

import org.springframework.ai.chat.messages.Message;
import org.springframework.util.Assert;

/**
 * The contract for storing and managing the memory of chat conversations.
 *
 * <p>
 * 中文导读：这是按 conversationId 管理对话上下文的策略接口。
 * 应用通过记忆 Advisor 在调用模型前读取消息、在调用后写入回答；接口本身不会拦截 ChatModel。
 * 例如 MessageWindowChatMemory 决定保留哪些消息，ChatMemoryRepository 决定消息存在哪里。
 * 因此“记忆”可能是有限窗口，不能直接当作完整聊天记录或持久化审计日志。
 *
 * <p>
 * conversationId 是应用提供的逻辑会话标识，同一个值会访问同一份历史。
 * 它不自动绑定用户身份或提供访问控制；同一用户的不同会话也需要使用不同标识才能彼此隔离。
 *
 * @author Christian Tzolov
 * @author Thomas Vitale
 * @since 1.0.0
 */
public interface ChatMemory {

	/**
	 * The key to retrieve the chat memory conversation id from the context.
	 * <p>
	 * ChatClient 使用时经 advisors 的 param 放入请求 context，记忆 Advisor 再按此键取值。
	 * 该值不是发给模型的用户文本，也不同于传给工具的 ToolContext。
	 */
	String CONVERSATION_ID = "chat_memory_conversation_id";

	/**
	 * Save the specified message in the chat memory for the specified conversation.
	 * <p>
	 * 单条入口统一委托给批量入口，使具体实现只需在一个地方实现窗口更新和存储策略。
	 */
	default void add(String conversationId, Message message) {
		Assert.hasText(conversationId, "conversationId cannot be null or empty");
		Assert.notNull(message, "message cannot be null");
		this.add(conversationId, List.of(message));
	}

	/**
	 * Save the specified messages in the chat memory for the specified conversation.
	 */
	void add(String conversationId, List<Message> messages);

	/**
	 * Get the messages in the chat memory for the specified conversation.
	 * <p>
	 * 返回当前策略保留下来的消息；是否复制列表、是否持久化以及并发保证由实现决定。
	 */
	List<Message> get(String conversationId);

	/**
	 * Clear the chat memory for the specified conversation.
	 */
	void clear(String conversationId);

}
