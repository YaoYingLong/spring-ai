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

package org.springframework.ai.model.tool;

import java.util.List;

import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.definition.ToolDefinition;

/**
 * Service responsible for managing the tool calling process for a chat model.
 *
 * <p>
 * 中文导读：把“向模型描述工具”和“执行模型请求的工具”分成两个阶段。
 * resolveToolDefinitions 从选项取出模型需要的工具定义；executeToolCalls 则消费一次模型响应，
 * 运行回调并产出包含 assistant 工具请求、tool 执行结果的新消息历史。
 *
 * <p>
 * 本接口不负责再次调用模型。ChatClient 的 ToolCallingAdvisor 使用返回历史继续下一轮，
 * 直接使用 ChatModel 的应用也可据此显式实现自己的工具循环、终止条件和消息存储策略。
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
public interface ToolCallingManager {

	/**
	 * Resolve the tool definitions from the model's tool calling options.
	 * <p>
	 * 只解析名称、描述和输入 Schema，不会调用工具。实际可执行回调仍保存在选项中供执行阶段查找。
	 */
	List<ToolDefinition> resolveToolDefinitions(ToolCallingChatOptions chatOptions);

	/**
	 * Execute the tool calls requested by the model.
	 * <p>
	 * prompt 提供已有消息、该轮有效工具及应用 ToolContext，chatResponse 提供模型产生的工具名、
	 * 调用 ID 和 JSON 参数。返回的 conversationHistory 用于续接对话，returnDirect 用于决定是否结束循环。
	 * 默认实现是同步执行；线程切换由调用方负责，方法返回后才具备完整的工具执行结果。
	 */
	ToolExecutionResult executeToolCalls(Prompt prompt, ChatResponse chatResponse);

	/**
	 * Create a default {@link ToolCallingManager} builder.
	 */
	static DefaultToolCallingManager.Builder builder() {
		return DefaultToolCallingManager.builder();
	}

}
