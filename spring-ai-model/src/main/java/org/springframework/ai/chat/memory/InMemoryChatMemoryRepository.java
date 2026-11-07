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
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.ai.chat.messages.Message;
import org.springframework.util.Assert;

/**
 * An in-memory implementation of {@link ChatMemoryRepository}.
 *
 * <p>
 * 中文导读：用 conversationId 作为 Map 的键，保存该会话当前的完整消息列表。
 * 本类不裁剪窗口、不计算 token，也不生成会话标识；这些职责分别在 ChatMemory 策略和应用层。
 * 数据只属于当前仓库实例，进程重启后丢失；这里没有 TTL、容量淘汰或跨实例同步。
 *
 * <p>
 * ConcurrentHashMap 保证单次 Map 操作的线程安全。
 * MessageWindowChatMemory 的“读取旧值、合并、覆盖”跨越多次操作，不能据此推导为同一会话的原子追加。
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
public final class InMemoryChatMemoryRepository implements ChatMemoryRepository {

	Map<String, List<Message>> chatMemoryStore = new ConcurrentHashMap<>();

	@Override
	public List<String> findConversationIds() {
		return new ArrayList<>(this.chatMemoryStore.keySet());
	}

	@Override
	public List<Message> findByConversationId(String conversationId) {
		Assert.hasText(conversationId, "conversationId cannot be null or empty");
		List<Message> messages = this.chatMemoryStore.get(conversationId);
		// 返回新的列表容器，调用方增删列表不会改写仓库；其中 Message 对象仍是共享引用。
		return messages != null ? new ArrayList<>(messages) : List.of();
	}

	@Override
	public void saveAll(String conversationId, List<Message> messages) {
		Assert.hasText(conversationId, "conversationId cannot be null or empty");
		Assert.notNull(messages, "messages cannot be null");
		Assert.noNullElements(messages, "messages cannot contain null elements");
		// saveAll 的语义是替换此会话的快照，而不是在旧记录后追加。
		// List.copyOf 固定列表结构，避免调用方随后修改原列表影响已保存的窗口。
		this.chatMemoryStore.put(conversationId, List.copyOf(messages));
	}

	@Override
	public void deleteByConversationId(String conversationId) {
		Assert.hasText(conversationId, "conversationId cannot be null or empty");
		this.chatMemoryStore.remove(conversationId);
	}

}
