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

import java.util.ArrayList;
import java.util.List;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;

import org.springframework.ai.chat.client.ChatClientMessageAggregator;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseAdvisor;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisorChain;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.util.Assert;

/**
 * Memory is retrieved added as a collection of messages to the prompt
 *
 * <p>
 * 中文导读：本类把 ChatMemory 接入 ChatClient 的 Advisor 链。
 * before 将历史 Message 放到本次 Prompt 前面，after 把模型输出写回同一会话。
 * 历史保留 USER、ASSISTANT、TOOL 等消息角色，不需要把整段聊天记录拼成一条文本提示词。
 *
 * <p>
 * 调用方通过 {@code advisors(a -> a.param(ChatMemory.CONVERSATION_ID, conversationId))}
 * 指定会话，BaseChatMemoryAdvisor 从请求 context 读取该值；当前实现要求它非空，没有默认会话 ID。
 * 默认 order 位于 ToolCallingAdvisor 之前，记忆包围整次工具循环，只写入本次输入和最终输出。
 * 若把记忆 Advisor 放到工具 Advisor 后面，则每轮模型调用都会经过这里，才能记录工具中间消息。
 *
 * @author Christian Tzolov
 * @author Mark Pollack
 * @author Thomas Vitale
 * @since 1.0.0
 */
public final class MessageChatMemoryAdvisor implements BaseChatMemoryAdvisor {

	private final ChatMemory chatMemory;

	private final int order;

	private final Scheduler scheduler;

	private MessageChatMemoryAdvisor(ChatMemory chatMemory, int order, Scheduler scheduler) {
		Assert.notNull(chatMemory, "chatMemory cannot be null");
		Assert.notNull(scheduler, "scheduler cannot be null");
		this.chatMemory = chatMemory;
		this.order = order;
		this.scheduler = scheduler;
	}

	@Override
	public int getOrder() {
		return this.order;
	}

	@Override
	public Scheduler getScheduler() {
		return this.scheduler;
	}

	@Override
	public ChatClientRequest before(ChatClientRequest chatClientRequest, AdvisorChain advisorChain) {
		// 会话路由来自 Advisor context；历史仓库不会从模型文本或 ToolContext 推断当前用户。
		String conversationId = getConversationId(chatClientRequest.context());

		// 1. Retrieve the chat memory for the current conversation.
		List<Message> memoryMessages = this.chatMemory.get(conversationId);

		// 2. Advise the request messages list.
		List<Message> promptMessages = chatClientRequest.prompt().getInstructions();
		List<Message> processedMessages = new ArrayList<>();
		// 工具循环可能已经携带完整历史；发现历史是 Prompt 中的连续子序列时不再重复前置。
		// 比较的是 Message.equals 和消息顺序，不是只按文本去重，也不会消除零散重复消息。
		if (!isMemoryAlreadyInPrompt(promptMessages, memoryMessages)) {
			processedMessages.addAll(memoryMessages);
		}
		processedMessages.addAll(promptMessages);

		// 2.1. Ensure system message, if present, appears first in the list.
		// 只移动遇到的第一条系统消息；系统指令的替换和保留策略仍由 ChatMemory 实现负责。
		for (int i = 0; i < processedMessages.size(); i++) {
			if (processedMessages.get(i) instanceof SystemMessage) {
				Message systemMessage = processedMessages.remove(i);
				processedMessages.add(0, systemMessage);
				break;
			}
		}

		// 3. Create a new request with the advised messages.
		ChatClientRequest processedChatClientRequest = chatClientRequest.mutate()
			.prompt(chatClientRequest.prompt().mutate().messages(processedMessages).build())
			.build();

		// 4. Add the new user message to the conversation memory.
		// 取最后一个 USER 或 TOOL response，而不是把增强后的整份 Prompt 再写回。
		// 嵌在工具循环内时，这里写入上一轮工具结果；普通调用则写入最新用户消息。
		// 写入发生在模型调用前，后续调用异常不会自动撤销这条输入。
		Message userMessage = processedChatClientRequest.prompt().getLastUserOrToolResponseMessage();
		this.chatMemory.add(conversationId, userMessage);

		return processedChatClientRequest;
	}

	private static boolean isMemoryAlreadyInPrompt(List<Message> promptMessages, List<Message> memoryMessages) {
		if (memoryMessages.isEmpty()) {
			return true;
		}
		if (promptMessages.size() < memoryMessages.size()) {
			return false;
		}
		for (int offset = 0; offset <= promptMessages.size() - memoryMessages.size(); offset++) {
			if (startsWith(promptMessages, memoryMessages, offset)) {
				return true;
			}
		}
		return false;
	}

	private static boolean startsWith(List<Message> messages, List<Message> prefix, int offset) {
		if (messages.size() - offset < prefix.size()) {
			return false;
		}
		for (int i = 0; i < prefix.size(); i++) {
			if (!messages.get(i + offset).equals(prefix.get(i))) {
				return false;
			}
		}
		return true;
	}

	@Override
	public ChatClientResponse after(ChatClientResponse chatClientResponse, AdvisorChain advisorChain) {
		// 保存每个 Generation 的完整 AssistantMessage，使工具调用等结构化部分随消息一起保留。
		List<Message> assistantMessages = new ArrayList<>();
		if (chatClientResponse.chatResponse() != null) {
			assistantMessages = chatClientResponse.chatResponse()
				.getResults()
				.stream()
				.map(g -> (Message) g.getOutput())
				.toList();
		}
		this.chatMemory.add(this.getConversationId(chatClientResponse.context()), assistantMessages);
		return chatClientResponse;
	}

	@Override
	public Flux<ChatClientResponse> adviseStream(ChatClientRequest chatClientRequest,
			StreamAdvisorChain streamAdvisorChain) {
		// Get the scheduler from BaseAdvisor
		Scheduler scheduler = this.getScheduler();

		// Process the request with the before method
		// 仓库接口是同步的，publishOn 让 before 在配置的 scheduler 上执行。
		// 流式响应仍逐块下发；聚合器仅在正常完成时调用一次 after，避免把每个文本片段当成独立回答。
		// 若上游报错或订阅取消，聚合完成回调不会保存一个看似完整的回答，已写入的输入则仍保留。
		return Mono.just(chatClientRequest)
			.publishOn(scheduler)
			.map(request -> this.before(request, streamAdvisorChain))
			.flatMapMany(streamAdvisorChain::nextStream)
			.transform(flux -> new ChatClientMessageAggregator().aggregateChatClientResponse(flux,
					response -> this.after(response, streamAdvisorChain)));
	}

	public static Builder builder(ChatMemory chatMemory) {
		return new Builder(chatMemory);
	}

	public static final class Builder {

		private int order = Advisor.DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER;

		private Scheduler scheduler = BaseAdvisor.DEFAULT_SCHEDULER;

		private final ChatMemory chatMemory;

		private Builder(ChatMemory chatMemory) {
			Assert.notNull(chatMemory, "chatMemory cannot be null");
			this.chatMemory = chatMemory;
		}

		/**
		 * Set the order.
		 * @param order the order
		 * @return the builder
		 */
		public Builder order(int order) {
			this.order = order;
			return this;
		}

		public Builder scheduler(Scheduler scheduler) {
			this.scheduler = scheduler;
			return this;
		}

		/**
		 * Build the advisor.
		 * @return the advisor
		 */
		public MessageChatMemoryAdvisor build() {
			return new MessageChatMemoryAdvisor(this.chatMemory, this.order, this.scheduler);
		}

	}

}
