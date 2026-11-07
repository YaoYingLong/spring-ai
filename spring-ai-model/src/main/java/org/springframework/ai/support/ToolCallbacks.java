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

package org.springframework.ai.support;

import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.method.MethodToolCallbackProvider;

/**
 * Provides {@link ToolCallback} instances for tools defined in different sources.
 *
 * <p>
 * 中文导读：这是把带 {@code @Tool} 方法的普通 Java 对象转换成工具回调的便捷入口。
 * ChatClient 的 tools 方法会先区分已有 ToolCallback、ToolCallbackProvider 和普通对象，
 * 只有普通对象才走这里；本方法本身不负责展开所有类型的工具容器。
 *
 * <p>
 * MethodToolCallbackProvider 扫描工具方法，并用 ToolDefinitions 从方法名、注解、
 * 参数类型和参数名生成模型可读的定义，再把 Method 与实际对象封装成 MethodToolCallback。
 * 这是注册阶段的元数据构建，不会立即调用业务方法；业务方法只在模型请求对应工具时执行。
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
public final class ToolCallbacks {

	private ToolCallbacks() {
	}

	/**
	 * 将给定对象中受支持的 {@code @Tool} 方法注册成回调。
	 * <p>
	 * Provider 会校验对象是否有工具方法以及工具名是否重复；AOP 代理的扫描使用目标类。
	 * 参数名同时决定 JSON Schema 的属性名和执行时的取值键，编译工具类时应保留 {@code -parameters}。
	 * @param sources 持有工具方法的对象实例
	 * @return 绑定方法定义与调用目标的工具回调
	 */
	public static ToolCallback[] from(Object... sources) {
		return MethodToolCallbackProvider.builder().toolObjects(sources).build().getToolCallbacks();
	}

}
