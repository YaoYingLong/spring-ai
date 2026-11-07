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

package org.springframework.ai.mcp;

import java.util.Map;

import io.modelcontextprotocol.client.McpAsyncClient;
import io.modelcontextprotocol.spec.McpError;
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest;
import io.modelcontextprotocol.spec.McpSchema.CallToolResult;
import io.modelcontextprotocol.spec.McpSchema.Tool;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jspecify.annotations.Nullable;

import org.springframework.ai.chat.model.ToolContext;
import org.springframework.ai.model.tool.internal.ToolCallReactiveContextHolder;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.ai.tool.execution.ToolExecutionException;
import org.springframework.ai.util.JsonHelper;
import org.springframework.util.Assert;
import org.springframework.util.StringUtils;

/**
 * Adapts MCP tools to Spring AI's {@link ToolCallback} interface with asynchronous
 * execution.
 * <p>
 * Bridges Model Context Protocol (MCP) tools with Spring AI's tool system, enabling
 * seamless integration of MCP tools in Spring AI applications.
 *
 * <p>
 * 中文导读：这里的 Async 描述所适配的 McpAsyncClient，ToolCallback 的 call 仍同步返回 String。
 * 因此本类订阅 MCP 客户端返回的 Mono，并用 block 等待完整结果；它不是一个返回 Mono 的工具 SPI。
 * ChatClient 的流式工具循环会在 boundedElastic 上执行此回调，直接调用者也需要考虑这个阻塞边界。
 *
 * <p>
 * 工具定义来自服务端的 Tool，模型选中适配名后，本类用服务端原名发起 MCP tools/call。
 * 客户端由外部创建和管理，本类不负责初始化或关闭它，也不会在每次工具调用后断开连接。
 * 外层聊天流的取消不能推导为远端操作已停止或已回滚；远端终止仍取决于客户端、传输与服务端。
 *
 * @author Christian Tzolov
 * @author YunKui Lu
 * @author Ilayaperumal Gopinathan
 */
public class AsyncMcpToolCallback implements ToolCallback {

	private static final JsonHelper jsonHelper = new JsonHelper();

	private static final Log logger = LogFactory.getLog(AsyncMcpToolCallback.class);

	private final McpAsyncClient mcpClient;

	private final Tool tool;

	private final String prefixedToolName;

	private final ToolContextToMcpMetaConverter toolContextToMcpMetaConverter;

	/**
	 * Creates an AsyncMcpToolCallback with default prefixed tool name.
	 * @param mcpClient the MCP client for tool execution
	 * @param tool the MCP tool to adapt
	 * @deprecated use {@link Builder} instead
	 */
	@Deprecated
	public AsyncMcpToolCallback(McpAsyncClient mcpClient, Tool tool) {
		this(mcpClient, tool, McpToolUtils.prefixedToolName(mcpClient.getClientInfo().name(),
				mcpClient.getClientInfo().title(), tool.name()), ToolContextToMcpMetaConverter.defaultConverter());
	}

	/**
	 * Creates an AsyncMcpToolCallback with specified parameters.
	 * @param mcpClient the MCP client for tool execution
	 * @param tool the MCP tool to adapt
	 * @param prefixedToolName the prefixed tool name for the tool definition
	 * @param toolContextToMcpMetaConverter converter for tool context to MCP metadata
	 */
	private AsyncMcpToolCallback(McpAsyncClient mcpClient, Tool tool, String prefixedToolName,
			ToolContextToMcpMetaConverter toolContextToMcpMetaConverter) {
		Assert.notNull(mcpClient, "MCP client must not be null");
		Assert.notNull(tool, "MCP tool must not be null");
		Assert.hasText(prefixedToolName, "Prefixed tool name must not be empty");
		Assert.notNull(toolContextToMcpMetaConverter, "ToolContextToMcpMetaConverter must not be null");

		this.mcpClient = mcpClient;
		this.tool = tool;
		this.prefixedToolName = prefixedToolName;
		this.toolContextToMcpMetaConverter = toolContextToMcpMetaConverter;
	}

	@Override
	public ToolDefinition getToolDefinition() {
		// 对模型暴露适配名和规范化的输入 Schema；真正发往 MCP 服务端的名称仍是 this.tool.name()。
		return McpToolUtils.createToolDefinition(this.prefixedToolName, this.tool);
	}

	public String getOriginalToolName() {
		return this.tool.name();
	}

	@Override
	public String call(String toolCallInput) {
		return this.call(toolCallInput, null);
	}

	@Override
	public String call(String toolCallInput, @Nullable ToolContext toolContext) {

		// Handle the possible null parameter situation in streaming mode.
		if (!StringUtils.hasText(toolCallInput)) {
			if (logger.isWarnEnabled()) {
				logger.warn("Tool call arguments are null or empty for MCP tool: " + this.tool.name()
						+ ". Using empty JSON object as default.");
			}
			toolCallInput = "{}";
		}

		// 模型提供的 JSON 参数是 MCP arguments；解析发生在 try 外，失败不会被下方 catch 统一包装。
		Map<String, Object> arguments = jsonHelper.fromJsonToMap(toolCallInput);

		CallToolResult response;
		try {
			// ToolContext 是应用提供的另一条数据通道，经转换器进入 MCP _meta。
			// 默认转换器会发送除 exchange 和 null 值以外的条目，并非只把它留在本地线程中。
			var mcpMeta = toolContext != null ? this.toolContextToMcpMetaConverter.convert(toolContext) : null;

			// Use the original tool name, not the prefixed one from getToolDefinition
			var request = CallToolRequest.builder(this.tool.name()).arguments(arguments).meta(mcpMeta).build();

			// Only map non-McpError exceptions to ToolExecutionException. McpError is a
			// protocol-level signal (e.g. URL elicitation) that must propagate as a hard
			// failure rather than being conveyed to the model as an error result.
			// block 会建立内部订阅，需要用 contextWrite 重新注入工具 Advisor 捕获的 Reactor Context。
			// 该 Context 用于 observation 等链路信息，与上面发送给服务端的 ToolContext/_meta 分属不同用途。
			// 本适配没有额外的取消令牌或远端回滚逻辑；等待和取消语义沿用底层 Mono/MCP 客户端。
			response = this.mcpClient.callTool(request).onErrorMap(e -> !(e instanceof McpError), e -> {
				logger.error("Exception while tool calling: ", e);
				return new ToolExecutionException(this.getToolDefinition(), e);
			}).contextWrite(ctx -> ctx.putAll(ToolCallReactiveContextHolder.getContext())).block();
		}
		catch (McpError ex) {
			// 协议级错误保持原样向上抛出；Manager 只捕获 ToolExecutionException，因此会终止本次交互。
			logger.error("Protocol error while calling tool: ", ex);
			// Since the tool calling manager only handles ToolExecutionException, this
			// bubbles up and fails the model interaction.
			throw ex;
		}
		catch (Exception ex) {
			logger.error("Exception while tool calling: ", ex);
			throw new ToolExecutionException(this.getToolDefinition(), ex);
		}
		Assert.notNull(response, "response was null");

		if (response.isError() != null && response.isError()) {
			// 收到了有效 MCP 响应但工具执行失败，可交由 Manager 的异常策略生成模型可读的错误结果。
			if (logger.isErrorEnabled()) {
				logger.error("Error calling tool: " + response.content());
			}
			throw new ToolExecutionException(this.getToolDefinition(),
					new IllegalStateException("Error calling tool: " + response.content()));
		}
		// 将内容块列表序列化成工具结果字符串；当前路径不读取 CallToolResult 的 structuredContent。
		return jsonHelper.toJson(response.content());
	}

	/**
	 * Creates a builder for constructing AsyncMcpToolCallback instances.
	 * @return a new builder
	 */
	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Builder for constructing AsyncMcpToolCallback instances.
	 */
	public static final class Builder {

		private @Nullable McpAsyncClient mcpClient;

		private @Nullable Tool tool;

		private @Nullable String prefixedToolName;

		private ToolContextToMcpMetaConverter toolContextToMcpMetaConverter = ToolContextToMcpMetaConverter
			.defaultConverter();

		/**
		 * Sets the MCP client for tool execution.
		 * @param mcpClient the MCP client (required)
		 * @return this builder
		 */
		public Builder mcpClient(McpAsyncClient mcpClient) {
			this.mcpClient = mcpClient;
			return this;
		}

		/**
		 * Sets the MCP tool to adapt.
		 * @param tool the MCP tool (required)
		 * @return this builder
		 */
		public Builder tool(Tool tool) {
			this.tool = tool;
			return this;
		}

		/**
		 * Sets the prefixed tool name for the tool definition.
		 * <p>
		 * Defaults to a generated name using the client and tool names.
		 * @param prefixedToolName the prefixed tool name
		 * @return this builder
		 */
		public Builder prefixedToolName(String prefixedToolName) {
			this.prefixedToolName = prefixedToolName;
			return this;
		}

		/**
		 * Sets the converter for tool context to MCP metadata transformation.
		 * <p>
		 * Defaults to {@link ToolContextToMcpMetaConverter#defaultConverter()}.
		 * @param toolContextToMcpMetaConverter the converter
		 * @return this builder
		 */
		public Builder toolContextToMcpMetaConverter(ToolContextToMcpMetaConverter toolContextToMcpMetaConverter) {
			Assert.notNull(toolContextToMcpMetaConverter, "ToolContextToMcpMetaConverter must not be null");
			this.toolContextToMcpMetaConverter = toolContextToMcpMetaConverter;
			return this;
		}

		/**
		 * Builds an AsyncMcpToolCallback with the configured parameters.
		 * @return a new AsyncMcpToolCallback
		 * @throws IllegalArgumentException if required parameters are missing
		 */
		public AsyncMcpToolCallback build() {
			Assert.notNull(this.mcpClient, "MCP client must not be null");
			Assert.notNull(this.tool, "MCP tool must not be null");

			// Apply defaults if not specified
			// 当前 builder 默认只格式化原工具名，跨 MCP 连接的去冲突前缀由 Provider 或调用方传入。
			if (this.prefixedToolName == null) {
				this.prefixedToolName = McpToolUtils.format(this.tool.name());
			}

			return new AsyncMcpToolCallback(this.mcpClient, this.tool, this.prefixedToolName,
					this.toolContextToMcpMetaConverter);
		}

	}

}
