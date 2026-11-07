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

package org.springframework.ai.tool;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jspecify.annotations.Nullable;

import org.springframework.ai.chat.model.ToolContext;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.ai.tool.metadata.ToolMetadata;

/**
 * Represents a tool whose execution can be triggered by an AI model.
 *
 * <p>
 * 中文导读：模型侧只看到 ToolDefinition 中的名称、用途和输入 JSON Schema；
 * 模型返回工具名称与参数后，ToolCallingManager 才在应用进程中找到对应回调并调用它。
 * 同一接口可包装 Java 方法、函数或远端 MCP 工具，业务执行入口统一为 call。
 *
 * <p>
 * call 是同步的字符串输入、字符串输出契约；它本身不创建线程，也没有流式结果、超时或取消参数。
 * 调度由上层负责，例如 ToolCallingAdvisor 的流式路径会把阻塞工具执行放到 boundedElastic。
 * ToolContext 是应用额外提供的运行上下文，与模型生成的 toolInput 分开传递。
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
public interface ToolCallback {

	Log logger = LogFactory.getLog(ToolCallback.class);

	/**
	 * Definition used by the AI model to determine when and how to call the tool.
	 * <p>
	 * 定义用于描述工具能力；读取定义不会执行工具，Schema 本身也不等于运行时参数校验器。
	 */
	ToolDefinition getToolDefinition();

	/**
	 * Metadata providing additional information on how to handle the tool.
	 * <p>
	 * 例如 returnDirect 告诉工具循环是否直接返回结果，这属于应用编排策略，不是模型输入参数。
	 */
	default ToolMetadata getToolMetadata() {
		return ToolMetadata.builder().build();
	}

	/**
	 * Execute tool with the given input and return the result to send back to the AI
	 * model.
	 */
	String call(String toolInput);

	/**
	 * Execute tool with the given input and context, and return the result to send back
	 * to the AI model.
	 * <p>
	 * 默认实现仅转调单参数 call；需要身份、租户或其他应用上下文的实现必须覆盖此重载。
	 * 上层提供了 context 并不代表任意 ToolCallback 都会使用它。
	 */
	default String call(String toolInput, @Nullable ToolContext toolContext) {
		if (toolContext != null && !toolContext.getContext().isEmpty()) {
			if (logger.isInfoEnabled()) {
				logger.info("By default the tool context is not used,  "
						+ "override the method 'call(String toolInput, ToolContext toolcontext)' to support the use of tool context."
						+ "Review the ToolCallback implementation for " + getToolDefinition().name());
			}
		}
		return call(toolInput);
	}

}
