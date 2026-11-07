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

import org.jspecify.annotations.Nullable;

import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.util.Assert;

/**
 * Represents a request processed by a {@link ChatClient} that ultimately is used to build
 * a {@link Prompt} to be sent to an AI model.
 * <p>
 * 中文导读：Advisor 链上的请求信封。prompt 保存有序消息及模型选项，context 保存
 * 链路本地的数据，例如会话 ID、检索文档和输出格式指令；context 不会整体发送给模型。
 * Advisor 通常通过 mutate() 构建下一版请求，并把它传给链上的下一个节点。
 * <p>
 * record 只固定两个字段的引用，不保证字段内部深度不可变。context 的 Map 可以修改，
 * copy()/mutate() 复制 Map 容器，但其中的业务对象仍共享；Prompt.copy() 也不是任意对象的深拷贝。
 *
 * @param prompt The prompt to be sent to the AI model
 * @param context The contextual data through the execution chain
 * @author Thomas Vitale
 * @since 1.0.0
 */
public record ChatClientRequest(Prompt prompt, Map<String, @Nullable Object> context) {

	public ChatClientRequest {
		Assert.notNull(prompt, "prompt cannot be null");
		Assert.notNull(context, "context cannot be null");
		Assert.noNullElements(context.keySet(), "context keys cannot be null");
	}

	public ChatClientRequest copy() {
		return new ChatClientRequest(this.prompt.copy(), new HashMap<>(this.context));
	}

	public Builder mutate() {
		return new Builder().prompt(this.prompt.copy()).context(new HashMap<>(this.context));
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private @Nullable Prompt prompt;

		private final Map<String, @Nullable Object> context = new HashMap<>();

		private Builder() {
		}

		public Builder prompt(Prompt prompt) {
			Assert.notNull(prompt, "prompt cannot be null");
			this.prompt = prompt;
			return this;
		}

		public Builder context(Map<String, ? extends @Nullable Object> context) {
			Assert.notNull(context, "context cannot be null");
			this.context.putAll(context);
			return this;
		}

		public Builder context(String key, @Nullable Object value) {
			Assert.notNull(key, "key cannot be null");
			this.context.put(key, value);
			return this;
		}

		public ChatClientRequest build() {
			Assert.state(this.prompt != null, "prompt cannot be null");
			return new ChatClientRequest(this.prompt, this.context);
		}

	}

}
