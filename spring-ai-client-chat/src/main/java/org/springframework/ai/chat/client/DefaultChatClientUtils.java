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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.Prompt.Builder;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;

/**
 * Utilities for supporting the {@link DefaultChatClient} implementation.
 * <p>
 * 中文导读：这是 fluent API 与模型领域对象之间的转换边界。
 * 输入是尚未渲染的请求 Spec，输出包含两部分：发给模型的 Prompt，以及供 Advisor
 * 在本地传递状态的 context。只有显式被 Advisor 写进 Prompt 的上下文才会进入模型请求。
 * 本方法由 call()/stream() 创建响应 Spec 时调用，尚不执行模型网络请求。
 *
 * @author Thomas Vitale
 * @author Sun Yuhan
 * @author Sebastien Deleuze
 * @since 1.0.0
 */
final class DefaultChatClientUtils {

	private DefaultChatClientUtils() {
		// prevents instantiation
	}

	static ChatClientRequest toChatClientRequest(DefaultChatClient.DefaultChatClientRequestSpec inputRequest) {
		Assert.notNull(inputRequest, "inputRequest cannot be null");

		/*
		 * ==========* MESSAGES * ==========
		 */

		List<Message> processedMessages = new ArrayList<>();

		// 消息顺序有语义：显式 system 文本在前，messages(...) 中的历史消息居中，当前 user 在后。
		// messages(...) 已是 Message 对象，原样加入，不会再次经过这里的模板渲染。
		// System Text => First in the list
		String processedSystemText = inputRequest.getSystemText();
		if (StringUtils.hasText(processedSystemText)) {
			// 只有提供变量时才调用渲染器；systemParams 与 userParams 分开，避免相互覆盖。
			if (!CollectionUtils.isEmpty(inputRequest.getSystemParams())) {
				processedSystemText = PromptTemplate.builder()
					.template(processedSystemText)
					.variables(inputRequest.getSystemParams())
					.renderer(inputRequest.getTemplateRenderer())
					.build()
					.render();
			}
			processedMessages.add(SystemMessage.builder()
				.text(processedSystemText)
				.metadata(inputRequest.getSystemMetadata())
				.build());
		}

		// Messages => In the middle of the list
		if (!CollectionUtils.isEmpty(inputRequest.getMessages())) {
			processedMessages.addAll(inputRequest.getMessages());
		}

		// User Text => Last in the list
		String processedUserText = inputRequest.getUserText();
		if (StringUtils.hasText(processedUserText)) {
			if (!CollectionUtils.isEmpty(inputRequest.getUserParams())) {
				processedUserText = PromptTemplate.builder()
					.template(processedUserText)
					.variables(inputRequest.getUserParams())
					.renderer(inputRequest.getTemplateRenderer())
					.build()
					.render();
			}
			processedMessages.add(UserMessage.builder()
				.text(processedUserText)
				.media(inputRequest.getMedia())
				.metadata(inputRequest.getUserMetadata())
				.build());
		}

		/*
		 * ==========* OPTIONS * ==========
		 */

		// 从具体模型的选项创建同类型 Builder，保留供应商特有字段和工具能力。
		// combineWith 以请求定制器的非 null 值覆盖默认值；没有指定的值沿用模型默认值。
		// 这是 ChatClient 路径的合并步骤，不能据此推断直接 ChatModel.call(prompt) 也会合并。
		ChatOptions.Builder<?> builder = inputRequest.getChatModel().getOptions().mutate();
		if (inputRequest.getOptionsCustomizer() != null) {
			builder = builder.combineWith(inputRequest.getOptionsCustomizer());
		}

		if (builder instanceof ToolCallingChatOptions.Builder<?> tbuilder) {

			// 已注册的回调直接使用；Provider 到构建本次请求时才展开，可反映动态工具集合。
			List<ToolCallback> toolCallbacks = new ArrayList<>(inputRequest.getToolCallbacks());
			for (var provider : inputRequest.getToolCallbackProviders()) {
				toolCallbacks.addAll(java.util.List.of(provider.getToolCallbacks()));
			}

			if (!toolCallbacks.isEmpty()) {
				ToolCallingChatOptions.validateToolCallbacks(toolCallbacks);
				tbuilder.toolCallbacks(toolCallbacks);
			}

			if (!inputRequest.getToolContext().isEmpty()) {
				// ToolContext 是交给本地工具执行的上下文，不是模型生成的工具 JSON 参数。
				tbuilder.toolContext(inputRequest.getToolContext());
			}

		}

		ChatOptions processedChatOptions = builder.build();

		/*
		 * ==========* REQUEST * ==========
		 */

		// advisorParams 进入请求 context，例如 conversationId、RAG 过滤条件、结构化输出开关。
		// 它与 userParams 的职责不同：后者已经在上面渲染为提示文本。
		Builder promptBuilder = Prompt.builder().messages(processedMessages).chatOptions(processedChatOptions);
		return ChatClientRequest.builder()
			.prompt(promptBuilder.build())
			.context(new ConcurrentHashMap<>(inputRequest.getAdvisorParams()))
			.build();
	}

}
