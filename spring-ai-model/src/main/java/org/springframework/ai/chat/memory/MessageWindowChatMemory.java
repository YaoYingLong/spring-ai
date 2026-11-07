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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.util.Assert;

/**
 * A chat memory implementation that maintains a message window of a specified size,
 * ensuring that the total number of messages does not exceed the specified limit. When
 * the number of messages exceeds the maximum size, older messages are evicted.
 * <p>
 * Messages of type {@link SystemMessage} are treated specially: if a new
 * {@link SystemMessage} is added, all previous {@link SystemMessage} instances are
 * removed from the memory. Also, if the total number of messages exceeds the limit, the
 * {@link SystemMessage} messages are preserved while evicting other types of messages.
 *
 * <p>
 * 中文导读：本类负责“记住什么”，仓库负责“存到哪里”。
 * 默认使用进程内仓库，窗口上限为 20 条消息；计数单位是 Message，不是 token，也不是问答轮数。
 * add 的完整流程是读出旧窗口、合并本次消息、裁剪，再用 saveAll 替换该会话的存储内容。
 *
 * <p>
 * 裁剪时优先保留 SystemMessage，其余消息按 USER 开始的一轮对话整体淘汰。
 * 这样在原始消息顺序完整的前提下，不会从某轮中部开始，留下缺少起始用户问题的回答或工具结果。
 * 这不是按 toolCallId 校验工具请求与响应的机制，也不会补齐调用方传入的不完整历史。
 *
 * @author Thomas Vitale
 * @author Ilayaperumal Gopinathan
 * @since 1.0.0
 */
public final class MessageWindowChatMemory implements ChatMemory {

	private static final int DEFAULT_MAX_MESSAGES = 20;

	private final ChatMemoryRepository chatMemoryRepository;

	private final int maxMessages;

	private MessageWindowChatMemory(ChatMemoryRepository chatMemoryRepository, int maxMessages) {
		Assert.notNull(chatMemoryRepository, "chatMemoryRepository cannot be null");
		Assert.isTrue(maxMessages > 0, "maxMessages must be greater than 0");
		this.chatMemoryRepository = chatMemoryRepository;
		this.maxMessages = maxMessages;
	}

	@Override
	public void add(String conversationId, List<Message> messages) {
		Assert.hasText(conversationId, "conversationId cannot be null or empty");
		Assert.notNull(messages, "messages cannot be null");
		Assert.noNullElements(messages, "messages cannot contain null elements");

		// 读、裁剪、写是三个独立步骤；仓库即使使用并发 Map，也不使本次 add 成为原子事务。
		// 同一 conversationId 的并发更新若需要严格保序，应由调用方或更高层协调。
		List<Message> memoryMessages = this.chatMemoryRepository.findByConversationId(conversationId);
		List<Message> processedMessages = process(memoryMessages, messages);
		this.chatMemoryRepository.saveAll(conversationId, processedMessages);
	}

	@Override
	public List<Message> get(String conversationId) {
		Assert.hasText(conversationId, "conversationId cannot be null or empty");
		return this.chatMemoryRepository.findByConversationId(conversationId);
	}

	@Override
	public void clear(String conversationId) {
		Assert.hasText(conversationId, "conversationId cannot be null or empty");
		this.chatMemoryRepository.deleteByConversationId(conversationId);
	}

	private List<Message> process(List<Message> memoryMessages, List<Message> newMessages) {
		List<Message> processedMessages = new ArrayList<>();

		// 只有出现与旧消息不相等的新系统指令，才淘汰所有旧系统指令。
		// HashSet 用于判断系统指令是否更新，并不对普通消息或整个历史做去重。
		Set<Message> memoryMessagesSet = new HashSet<>(memoryMessages);
		boolean hasNewSystemMessage = newMessages.stream()
			.filter(SystemMessage.class::isInstance)
			.anyMatch(message -> !memoryMessagesSet.contains(message));

		memoryMessages.stream()
			.filter(message -> !(hasNewSystemMessage && message instanceof SystemMessage))
			.forEach(processedMessages::add);

		processedMessages.addAll(newMessages);

		// 未超限时完整保留消息顺序；本类不会主动把 SystemMessage 移到开头。
		if (processedMessages.size() <= this.maxMessages) {
			return processedMessages;
		}

		// Collect the indices of non-system messages; SystemMessages are always
		// preserved.
		List<Integer> nonSystemIndices = new ArrayList<>();
		for (int i = 0; i < processedMessages.size(); i++) {
			if (!(processedMessages.get(i) instanceof SystemMessage)) {
				nonSystemIndices.add(i);
			}
		}

		// Raw cut: the number of non-system messages that must be removed to fit within
		// maxMessages. This index into nonSystemIndices is where the kept window would
		// start based on count alone.
		int cutIndex = processedMessages.size() - this.maxMessages;

		// 例如 [U1, A1, U2, A2] 上限为 3，按条数会从 A1 开始；找到后续 U2 后只保留 [U2, A2]。
		// 工具轮中的 assistant(tool_calls) 和 tool response 也随所属轮一起淘汰，窗口可能小于上限。
		// Snap the cut forward to the nearest USER message so the kept window always
		// starts at a complete turn. This prevents keeping an assistant reply or tool
		// result without the user message that originated its turn.
		while (cutIndex < nonSystemIndices.size()
				&& processedMessages.get(nonSystemIndices.get(cutIndex)).getMessageType() != MessageType.USER) {
			cutIndex++;
		}
		// 若剩余位置没有 USER，全部普通消息都会被移除；系统消息数本身超限时仍全部保留。
		cutIndex = Math.min(cutIndex, nonSystemIndices.size());

		Set<Integer> removeIndices = new HashSet<>(nonSystemIndices.subList(0, cutIndex));
		List<Message> trimmedMessages = new ArrayList<>();
		for (int i = 0; i < processedMessages.size(); i++) {
			if (!removeIndices.contains(i)) {
				trimmedMessages.add(processedMessages.get(i));
			}
		}
		return trimmedMessages;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		// 默认仓库跟随这个 builder 创建；若希望多个 ChatMemory 实例共享历史，需注入共享仓库。
		private ChatMemoryRepository chatMemoryRepository = new InMemoryChatMemoryRepository();

		private int maxMessages = DEFAULT_MAX_MESSAGES;

		private Builder() {
		}

		public Builder chatMemoryRepository(ChatMemoryRepository chatMemoryRepository) {
			this.chatMemoryRepository = chatMemoryRepository;
			return this;
		}

		public Builder maxMessages(int maxMessages) {
			this.maxMessages = maxMessages;
			return this;
		}

		public MessageWindowChatMemory build() {
			return new MessageWindowChatMemory(this.chatMemoryRepository, this.maxMessages);
		}

	}

}
