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

package org.springframework.ai.model.chat.memory.autoconfigure;

import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.ChatMemoryRepository;
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;

/**
 * Auto-configuration for {@link ChatMemory}.
 * <p>
 * 中文导读：把“消息如何保留”的 ChatMemory 策略与“消息存到哪里”的 ChatMemoryRepository 分开装配。
 * 类路径具备这两个接口时生效，各 Bean 在没有同类替代品时才提供默认实现。
 * <p>
 * 默认组合是 MessageWindowChatMemory 加进程内仓库；应用提供数据库仓库后，
 * 默认窗口策略仍可使用该仓库，应用也可以直接提供自己的 ChatMemory 替换策略。
 * 本配置只提供记忆组件，不会自动给 ChatClient 安装记忆 Advisor，也不负责生成 conversationId。
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
@AutoConfiguration
@ConditionalOnClass({ ChatMemory.class, ChatMemoryRepository.class })
public class ChatMemoryAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean
	ChatMemoryRepository chatMemoryRepository() {
		// 进程内默认仓库随应用实例生命周期存在；跨实例共享或重启后保留需要替换仓库实现。
		return new InMemoryChatMemoryRepository();
	}

	@Bean
	@ConditionalOnMissingBean
	ChatMemory chatMemory(ChatMemoryRepository chatMemoryRepository) {
		// 注入容器最终选中的仓库；未指定 maxMessages，使用窗口实现默认的 20 条消息配置。
		// 窗口按消息条数而非 token 数配置，System/工具消息的保留规则由 MessageWindowChatMemory 处理。
		return MessageWindowChatMemory.builder().chatMemoryRepository(chatMemoryRepository).build();
	}

}
