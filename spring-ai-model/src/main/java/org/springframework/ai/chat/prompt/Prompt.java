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

package org.springframework.ai.chat.prompt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

import org.jspecify.annotations.Nullable;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.model.ModelRequest;
import org.springframework.util.Assert;

/**
 * The Prompt class represents a prompt used in AI model requests. A prompt consists of
 * one or more messages and additional chat options.
 * <p>
 * 中文导读：Prompt 是模型调用的完整输入，不只是一个提示词字符串。
 * messages 保留 system/user/assistant/tool 的角色和顺序；chatOptions 保存模型名、
 * 生成参数以及具体选项类型所支持的工具、结构化输出配置。
 * new Prompt(String) 只是创建单条 UserMessage 的便捷写法。
 * <p>
 * 记忆 Advisor 通过增加历史消息实现多轮对话，RAG Advisor 通过改写用户消息增加检索上下文，
 * 工具循环通过追加 AssistantMessage 和 ToolResponseMessage 让模型看到工具执行结果。
 * 这些能力最终都汇合为本类，再由 ChatModel 转成供应商协议。
 * <p>
 * 本类没有自动防御性复制所有输入：构造器保留传入 List，copy()/mutate() 复制消息时
 * 仍复用 chatOptions。应按具体复制边界理解对象共享，不能仅凭 final 字段判断深度不可变。
 *
 * @author Mark Pollack
 * @author luocongqiu
 * @author Thomas Vitale
 * @author Sebastien Deleuze
 * @author guan xu
 */
public class Prompt implements ModelRequest<List<Message>> {

	private final List<Message> messages;

	private final @Nullable ChatOptions chatOptions;

	public Prompt(String contents) {
		this(new UserMessage(contents));
	}

	public Prompt(Message message) {
		this(Collections.singletonList(message));
	}

	public Prompt(List<Message> messages) {
		this(messages, null);
	}

	public Prompt(Message... messages) {
		this(Arrays.asList(messages), null);
	}

	public Prompt(String contents, @Nullable ChatOptions chatOptions) {
		this(new UserMessage(contents), chatOptions);
	}

	public Prompt(Message message, @Nullable ChatOptions chatOptions) {
		this(Collections.singletonList(message), chatOptions);
	}

	public Prompt(List<Message> messages, @Nullable ChatOptions chatOptions) {
		Assert.notNull(messages, "messages cannot be null");
		Assert.noNullElements(messages, "messages cannot contain null elements");
		this.messages = messages;
		this.chatOptions = chatOptions;
	}

	public String getContents() {
		// 便捷文本视图会丢失角色、工具和多模态结构；模型适配器应读取 getInstructions()。
		StringBuilder sb = new StringBuilder();
		for (Message message : getInstructions()) {
			sb.append(message.getText());
		}
		return sb.toString();
	}

	@Override
	public @Nullable ChatOptions getOptions() {
		return this.chatOptions;
	}

	@Override
	public List<Message> getInstructions() {
		return this.messages;
	}

	/**
	 * Get the first system message in the prompt. If no system message is found, an empty
	 * SystemMessage is returned.
	 */
	public SystemMessage getSystemMessage() {
		for (int i = 0; i <= this.messages.size() - 1; i++) {
			Message message = this.messages.get(i);
			if (message instanceof SystemMessage systemMessage) {
				return systemMessage;
			}
		}
		return new SystemMessage("");
	}

	/**
	 * Get the last user message in the prompt. If no user message is found, an empty
	 * UserMessage is returned.
	 */
	public UserMessage getUserMessage() {
		// 从后向前找当前问题。RAG 等组件通常只增强这一条，避免把每一条历史问题都当成本轮输入。
		for (int i = this.messages.size() - 1; i >= 0; i--) {
			Message message = this.messages.get(i);
			if (message instanceof UserMessage userMessage) {
				return userMessage;
			}
		}
		return new UserMessage("");
	}

	/**
	 * Get the last user or tool response message in the prompt. If no user or tool
	 * response message is found, an empty UserMessage is returned.
	 */
	public Message getLastUserOrToolResponseMessage() {
		// 工具循环下一轮的最新输入可能是工具结果，不能只取最后一条 UserMessage。
		for (int i = this.messages.size() - 1; i >= 0; i--) {
			Message message = this.messages.get(i);
			if (message instanceof UserMessage || message instanceof ToolResponseMessage) {
				return message;
			}
		}
		return new UserMessage("");
	}

	/**
	 * Get all system messages in the prompt.
	 * @return a list of all system messages in the prompt
	 */
	public List<SystemMessage> getSystemMessages() {
		List<SystemMessage> systemMessages = new ArrayList<>();
		for (Message message : this.messages) {
			if (message instanceof SystemMessage systemMessage) {
				systemMessages.add(systemMessage);
			}
		}
		return systemMessages;
	}

	/**
	 * Get all user messages in the prompt.
	 * @return a list of all user messages in the prompt
	 */
	public List<UserMessage> getUserMessages() {
		List<UserMessage> userMessages = new ArrayList<>();
		for (Message message : this.messages) {
			if (message instanceof UserMessage userMessage) {
				userMessages.add(userMessage);
			}
		}
		return userMessages;
	}

	@Override
	public String toString() {
		return "Prompt{" + "messages=" + this.messages + ", modelOptions=" + this.chatOptions + '}';
	}

	@Override
	public boolean equals(@Nullable Object o) {
		if (this == o) {
			return true;
		}
		if (!(o instanceof Prompt prompt)) {
			return false;
		}
		return Objects.equals(this.messages, prompt.messages) && Objects.equals(this.chatOptions, prompt.chatOptions);
	}

	@Override
	public int hashCode() {
		return Objects.hash(this.messages, this.chatOptions);
	}

	public Prompt copy() {
		// 支持的标准消息按各自 copy 规则复制；选项引用继续共享，不是整个请求对象图的深拷贝。
		return new Prompt(instructionsCopy(), this.chatOptions);
	}

	private List<Message> instructionsCopy() {
		List<Message> messagesCopy = new ArrayList<>();
		this.messages.forEach(message -> {
			if (message instanceof UserMessage userMessage) {
				messagesCopy.add(userMessage.copy());
			}
			else if (message instanceof SystemMessage systemMessage) {
				messagesCopy.add(systemMessage.copy());
			}
			else if (message instanceof AssistantMessage assistantMessage) {
				messagesCopy.add(assistantMessage.copy());
			}
			else if (message instanceof ToolResponseMessage toolResponseMessage) {
				messagesCopy.add(ToolResponseMessage.builder()
					.responses(new ArrayList<>(toolResponseMessage.getResponses()))
					.metadata(new HashMap<>(toolResponseMessage.getMetadata()))
					.build());
			}
			else {
				throw new IllegalArgumentException("Unsupported message type: " + message.getClass().getName());
			}
		});

		return messagesCopy;
	}

	/**
	 * Augments the first system message in the prompt with the provided function. If no
	 * system message is found, a new one is created with the provided text.
	 * @return a new {@link Prompt} instance with the augmented system message.
	 */
	public Prompt augmentSystemMessage(Function<SystemMessage, SystemMessage> systemMessageAugmenter) {
		// 创建新 List，仅替换第一条 system；未匹配时在开头加入，不原地改写当前消息列表。
		var messagesCopy = new ArrayList<>(this.messages);
		boolean found = false;
		for (int i = 0; i < messagesCopy.size(); i++) {
			Message message = messagesCopy.get(i);
			if (message instanceof SystemMessage systemMessage) {
				messagesCopy.set(i, systemMessageAugmenter.apply(systemMessage));
				found = true;
				break;
			}
		}
		if (!found) {
			// If no system message is found, create a new one with the provided text
			// and add it as the first item in the list.
			messagesCopy.add(0, systemMessageAugmenter.apply(new SystemMessage("")));
		}
		return new Prompt(messagesCopy, this.chatOptions);
	}

	/**
	 * Augments the last system message in the prompt with the provided text. If no system
	 * message is found, a new one is created with the provided text.
	 * @return a new {@link Prompt} instance with the augmented system message.
	 */
	public Prompt augmentSystemMessage(String newSystemText) {
		return augmentSystemMessage(systemMessage -> systemMessage.mutate().text(newSystemText).build());
	}

	/**
	 * Augments the last user message in the prompt with the provided function. If no user
	 * message is found, a new one is created with the provided text.
	 * @return a new {@link Prompt} instance with the augmented user message.
	 */
	public Prompt augmentUserMessage(Function<UserMessage, UserMessage> userMessageAugmenter) {
		// 倒序替换最后一条 user，保留其他历史消息和原选项；lambda 的参数是匹配到的原消息。
		// 注意当前实现的补建分支在循环内：空消息列表不会执行该分支。
		var messagesCopy = new ArrayList<>(this.messages);
		for (int i = messagesCopy.size() - 1; i >= 0; i--) {
			Message message = messagesCopy.get(i);
			if (message instanceof UserMessage userMessage) {
				messagesCopy.set(i, userMessageAugmenter.apply(userMessage));
				break;
			}
			if (i == 0) {
				messagesCopy.add(userMessageAugmenter.apply(new UserMessage("")));
			}
		}

		return new Prompt(messagesCopy, this.chatOptions);
	}

	/**
	 * Augments the last user message in the prompt with the provided text. If no user
	 * message is found, a new one is created with the provided text.
	 * @return a new {@link Prompt} instance with the augmented user message.
	 */
	public Prompt augmentUserMessage(String newUserText) {
		return augmentUserMessage(userMessage -> userMessage.mutate().text(newUserText).build());
	}

	public Builder mutate() {
		Builder builder = new Builder().messages(instructionsCopy());
		if (this.chatOptions != null) {
			builder.chatOptions(this.chatOptions);
		}
		return builder;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private @Nullable List<Message> messages;

		private @Nullable ChatOptions chatOptions;

		public Builder content(@Nullable String content) {
			this.messages = List.of(new UserMessage(content));
			return this;
		}

		public Builder messages(Message @Nullable ... messages) {
			if (messages != null) {
				this.messages = Arrays.asList(messages);
			}
			return this;
		}

		public Builder messages(List<Message> messages) {
			this.messages = messages;
			return this;
		}

		public Builder chatOptions(ChatOptions chatOptions) {
			this.chatOptions = chatOptions;
			return this;
		}

		public Prompt build() {
			Assert.state(this.messages != null, "either messages or content needs to be set");
			return new Prompt(this.messages, this.chatOptions);
		}

	}

}
