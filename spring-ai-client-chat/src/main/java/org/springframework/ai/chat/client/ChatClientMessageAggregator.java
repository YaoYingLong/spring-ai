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

package org.springframework.ai.chat.client;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import reactor.core.publisher.Flux;

import org.springframework.ai.chat.model.MessageAggregator;

/**
 * Helper that for streaming chat responses, aggregate the chat response messages into a
 * single AssistantMessage. Job is performed in parallel to the chat response processing.
 * <p>
 * 中文导读：为模型层的 MessageAggregator 补上 ChatClient context 的传播。
 * 每个片段的上下文合并到本次聚合 Map；正常完成后，完整模型响应与累计上下文一起
 * 交给 aggregationHandler，供观测或记忆组件使用。
 * <p>
 * 这里的旁路聚合使用同一次订阅中的操作符，没有另外启动后台线程或再次请求模型。
 * 下游仍逐片收到响应；聚合结果经回调交付，不额外作为最后一个片段发送。
 *
 * @author Christian Tzolov
 * @author Alexandros Pappas
 * @author Thomas Vitale
 * @since 1.0.0
 */
public class ChatClientMessageAggregator {

	@SuppressWarnings("NullAway") // https://github.com/uber/NullAway/issues/1350
	public Flux<ChatClientResponse> aggregateChatClientResponse(Flux<ChatClientResponse> chatClientResponses,
			Consumer<ChatClientResponse> aggregationHandler) {

		AtomicReference<Map<String, Object>> context = new AtomicReference<>(new HashMap<>());

		return new MessageAggregator().aggregate(chatClientResponses.mapNotNull(chatClientResponse -> {
			context.get().putAll(chatClientResponse.context());
			return chatClientResponse.chatResponse();
		}), aggregatedChatResponse -> {
			ChatClientResponse aggregatedChatClientResponse = ChatClientResponse.builder()
				.chatResponse(aggregatedChatResponse)
				.context(context.get())
				.build();
			aggregationHandler.accept(aggregatedChatClientResponse);
		}).map(chatResponse -> ChatClientResponse.builder().chatResponse(chatResponse).context(context.get()).build());
	}

}
