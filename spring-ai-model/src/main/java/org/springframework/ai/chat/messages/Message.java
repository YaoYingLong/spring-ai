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

package org.springframework.ai.chat.messages;

import org.springframework.ai.content.Content;
import org.springframework.ai.content.Media;

/**
 * The Message interface represents a message that can be sent or received in a chat
 * application. Messages can have content, media attachments, properties, and message
 * types.
 * <p>
 * 中文导读：消息角色是模型协议的一部分，不能把所有消息直接拼成一个字符串替代。
 * SystemMessage 表达规则，UserMessage 表达输入，AssistantMessage 表达模型生成内容
 * 或工具调用意图，ToolResponseMessage 表达工具执行结果，并通过调用 ID 对应工具请求。
 * <p>
 * 当前消息实现还可以承载文本、媒体、推理及工具调用等内容部分。
 * getText() 是面向纯文本的便利视图；阅读多模态和工具链时应继续查看具体消息的 parts，
 * 以及供应商适配器如何将角色和内容映射到 API。
 *
 * @see Media
 * @see MessageType
 */
public interface Message extends Content {

	/**
	 * Get the message type.
	 * @return the message type
	 */
	MessageType getMessageType();

}
