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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationRegistry;
import io.micrometer.observation.contextpropagation.ObservationThreadLocalAccessor;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.ToolContext;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.internal.ToolCallReactiveContextHolder;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.ai.tool.execution.DefaultToolExecutionExceptionProcessor;
import org.springframework.ai.tool.execution.ToolExecutionException;
import org.springframework.ai.tool.execution.ToolExecutionExceptionProcessor;
import org.springframework.ai.tool.observation.DefaultToolCallingObservationConvention;
import org.springframework.ai.tool.observation.ToolCallingObservationContext;
import org.springframework.ai.tool.observation.ToolCallingObservationConvention;
import org.springframework.ai.tool.observation.ToolCallingObservationDocumentation;
import org.springframework.ai.tool.resolution.DelegatingToolCallbackResolver;
import org.springframework.ai.tool.resolution.ToolCallbackResolver;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;

/**
 * Default implementation of {@link ToolCallingManager}.
 *
 * <p>
 * 中文导读：本类处理一次模型响应中的工具调用批次，不持有跨轮的可变计数，也不调用模型。
 * 核心路径是选择包含工具请求的候选回答、解析回调、检查限额、逐个执行并组装 ToolResponseMessage。
 * ToolCallingAdvisor 再把“旧历史 + assistant 工具请求 + tool 结果”传给下一轮模型。
 *
 * <p>
 * 执行是当前线程上的串行 for 循环；即使模型一次返回多个工具调用，也不代表本实现并行运行它们。
 * ChatClient 同步路径在调用线程执行，流式路径由工具 Advisor 切到 boundedElastic 后再进入这里。
 * 单个工具的副作用已经发生时，后续工具失败或触发限额不会自动回滚先前执行。
 *
 * @author Thomas Vitale
 * @author chabinhwang
 * @author Christian Tzolov
 * @author Dimitar Proynov
 * @since 1.0.0
 */
public final class DefaultToolCallingManager implements ToolCallingManager {

	private static final Log logger = LogFactory.getLog(DefaultToolCallingManager.class);

	// @formatter:off

	private static final ObservationRegistry DEFAULT_OBSERVATION_REGISTRY
			= ObservationRegistry.NOOP;

	private static final ToolCallingObservationConvention DEFAULT_OBSERVATION_CONVENTION
			= new DefaultToolCallingObservationConvention();

	private static final ToolCallbackResolver DEFAULT_TOOL_CALLBACK_RESOLVER
			= new DelegatingToolCallbackResolver(List.of());

	private static final ToolExecutionExceptionProcessor DEFAULT_TOOL_EXECUTION_EXCEPTION_PROCESSOR
			= DefaultToolExecutionExceptionProcessor.builder().build();

	private static final boolean DEFAULT_RESOLUTION_FALLBACK_ENABLED = false;

	private static final String POSSIBLE_LLM_TOOL_NAME_CHANGE_WARNING_START = "LLM may have adapted the tool name '";
	private static final String POSSIBLE_LLM_TOOL_NAME_CHANGE_WARNING_END
			= "', especially if the name was truncated due to length limits. If this is the case, you can customize the prefixing and processing logic using McpToolNamePrefixGenerator";

	// @formatter:on

	/**
	 * Default cap applied to every tool name unless overridden via
	 * {@link Builder#maxCallsPerTool(String, int)}, exempted via
	 * {@link Builder#excludeToolFromLimit(String)}, or disabled entirely via
	 * {@link Builder#unlimitedCallsPerTool()}. Generous enough for legitimate iterative
	 * tool use (pagination, retries) while still catching a runaway loop long before it
	 * becomes expensive.
	 */
	public static final int DEFAULT_MAX_CALLS_PER_TOOL = 40;

	/**
	 * Default cap on the total number of tool calls, across all tools combined, allowed
	 * within a turn, unless disabled entirely via
	 * {@link Builder#unlimitedTotalToolCalls()}.
	 */
	public static final int DEFAULT_MAX_TOTAL_TOOL_CALLS = 150;

	private final ObservationRegistry observationRegistry;

	private final ToolCallbackResolver toolCallbackResolver;

	private final ToolExecutionExceptionProcessor toolExecutionExceptionProcessor;

	private final ToolCallLimits toolCallLimits;

	/**
	 * Whether a call to a tool that is not part of the request's tool callbacks may still
	 * be executed by resolving it through the {@link ToolCallbackResolver}. When
	 * {@code false} (the default), the tools attached to the request are the only tools
	 * that can be executed.
	 */
	private final boolean resolutionFallbackEnabled;

	private ToolCallingObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	public DefaultToolCallingManager(ObservationRegistry observationRegistry, ToolCallbackResolver toolCallbackResolver,
			ToolExecutionExceptionProcessor toolExecutionExceptionProcessor) {
		this(observationRegistry, toolCallbackResolver, toolExecutionExceptionProcessor,
				new ToolCallLimits(DEFAULT_MAX_CALLS_PER_TOOL, Map.of(), Set.of(), DEFAULT_MAX_TOTAL_TOOL_CALLS,
						ToolCallLimitBehavior.THROW),
				DEFAULT_RESOLUTION_FALLBACK_ENABLED);
	}

	DefaultToolCallingManager(ObservationRegistry observationRegistry, ToolCallbackResolver toolCallbackResolver,
			ToolExecutionExceptionProcessor toolExecutionExceptionProcessor, ToolCallLimits toolCallLimits) {
		this(observationRegistry, toolCallbackResolver, toolExecutionExceptionProcessor, toolCallLimits,
				DEFAULT_RESOLUTION_FALLBACK_ENABLED);
	}

	/**
	 * Create a {@link DefaultToolCallingManager}.
	 * @param observationRegistry the observation registry
	 * @param toolCallbackResolver the resolver consulted for name-based tool resolution
	 * @param toolExecutionExceptionProcessor the processor for tool execution exceptions
	 * @param toolCallLimits the limits applied to tool calls within a turn
	 * @param resolutionFallbackEnabled whether tools that are not part of the request's
	 * tool callbacks may be resolved and executed via the {@code toolCallbackResolver};
	 * defaults to {@code false} so that the request's tool callbacks bound execution
	 * @since 2.0.1
	 */
	DefaultToolCallingManager(ObservationRegistry observationRegistry, ToolCallbackResolver toolCallbackResolver,
			ToolExecutionExceptionProcessor toolExecutionExceptionProcessor, ToolCallLimits toolCallLimits,
			boolean resolutionFallbackEnabled) {
		Assert.notNull(observationRegistry, "observationRegistry cannot be null");
		Assert.notNull(toolCallbackResolver, "toolCallbackResolver cannot be null");
		Assert.notNull(toolExecutionExceptionProcessor, "toolCallExceptionConverter cannot be null");
		Assert.notNull(toolCallLimits, "toolCallLimits cannot be null");

		this.observationRegistry = observationRegistry;
		this.toolCallbackResolver = toolCallbackResolver;
		this.toolExecutionExceptionProcessor = toolExecutionExceptionProcessor;
		this.toolCallLimits = toolCallLimits;
		this.resolutionFallbackEnabled = resolutionFallbackEnabled;
	}

	@Override
	public List<ToolDefinition> resolveToolDefinitions(ToolCallingChatOptions chatOptions) {
		Assert.notNull(chatOptions, "chatOptions cannot be null");

		List<ToolCallback> toolCallbacks = new ArrayList<>(
				!CollectionUtils.isEmpty(chatOptions.getToolCallbacks()) ? chatOptions.getToolCallbacks() : List.of());

		// 只暴露请求选项中显式携带的定义；这里不会遍历 resolver 找出其他可用工具。
		return toolCallbacks.stream().map(ToolCallback::getToolDefinition).toList();
	}

	@Override
	public ToolExecutionResult executeToolCalls(Prompt prompt, ChatResponse chatResponse) {
		Assert.notNull(prompt, "prompt cannot be null");
		Assert.notNull(chatResponse, "chatResponse cannot be null");

		// 一个响应可有多个候选 Generation，本实现只处理首个含 tool calls 的候选，不合并所有候选。
		Optional<Generation> toolCallGeneration = chatResponse.getResults()
			.stream()
			.filter(g -> !CollectionUtils.isEmpty(g.getOutput().getToolCalls()))
			.findFirst();

		if (toolCallGeneration.isEmpty()) {
			throw new IllegalStateException("No tool call requested by the chat model");
		}

		AssistantMessage assistantMessage = toolCallGeneration.get().getOutput();

		// ToolContext 来自 Prompt 选项中的应用数据，工具参数则来自 assistantMessage，两者不混合。
		ToolContext toolContext = buildToolContext(prompt, assistantMessage);

		InternalToolExecutionResult internalToolExecutionResult = executeToolCall(prompt, assistantMessage,
				toolContext);

		List<Message> conversationHistory = buildConversationHistoryAfterToolExecution(prompt.getInstructions(),
				assistantMessage, internalToolExecutionResult.toolResponseMessage());

		return ToolExecutionResult.builder()
			.conversationHistory(conversationHistory)
			.returnDirect(internalToolExecutionResult.returnDirect())
			.build();
	}

	private static ToolContext buildToolContext(Prompt prompt, AssistantMessage assistantMessage) {
		Map<String, Object> toolContextMap = Map.of();

		if (prompt.getOptions() instanceof ToolCallingChatOptions toolCallingChatOptions
				&& !CollectionUtils.isEmpty(toolCallingChatOptions.getToolContext())) {
			// 复制 Map 容器；本实现不会把对话历史或 assistantMessage 自动注入到 ToolContext 中。
			toolContextMap = new HashMap<>(toolCallingChatOptions.getToolContext());
		}

		return new ToolContext(toolContextMap);
	}

	/**
	 * Execute the tool call and return the response message.
	 */
	private InternalToolExecutionResult executeToolCall(Prompt prompt, AssistantMessage assistantMessage,
			ToolContext toolContext) {
		List<ToolCallback> toolCallbacks = List.of();
		if (prompt.getOptions() instanceof ToolCallingChatOptions toolCallingChatOptions) {
			if (!CollectionUtils.isEmpty(toolCallingChatOptions.getToolCallbacks())) {
				toolCallbacks = toolCallingChatOptions.getToolCallbacks();
			}
		}

		List<ToolResponseMessage.ToolResponse> toolResponses = new ArrayList<>();

		Boolean returnDirect = null;

		// 从最后一条 UserMessage 开始统计已存在的工具结果，避免把之前用户轮次的调用计入本轮限额。
		// Advisor 即使裁剪向后续链转发的 Prompt，也会单独保留完整本轮历史供此处重建计数。
		Map<String, Integer> toolCallCounts = ToolCallLimits.countPriorToolCalls(prompt.getInstructions());
		int totalToolCallCount = toolCallCounts.values().stream().mapToInt(Integer::intValue).sum();

		for (AssistantMessage.ToolCall toolCall : assistantMessage.getToolCalls()) {

			if (logger.isDebugEnabled()) {
				logger.debug("Executing tool call: " + toolCall.name());
			}

			String toolName = toolCall.name();

			totalToolCallCount++;
			int toolCallCount = toolCallCounts.merge(toolName, 1, Integer::sum);

			ToolCallLimits.Breach limitBreach = this.toolCallLimits.check(toolName, toolCallCount, totalToolCallCount);
			if (limitBreach != null) {
				// 先判断再执行：超限调用不会触发回调。默认每个工具 40 次、所有工具合计 150 次。
				// THROW 携带已完成部分并中断批次；RETURN_ERROR_RESPONSE 记录错误后继续处理余下调用。
				toolResponses.add(new ToolResponseMessage.ToolResponse(toolCall.id(), toolName, limitBreach.message()));

				if (this.toolCallLimits.onLimitExceeded() == ToolCallLimitBehavior.THROW) {
					ToolResponseMessage partialToolResponseMessage = ToolResponseMessage.builder()
						.responses(toolResponses)
						.build();
					List<Message> partialConversationHistory = buildConversationHistoryAfterToolExecution(
							prompt.getInstructions(), assistantMessage, partialToolResponseMessage);
					ToolExecutionResult partialToolExecutionResult = ToolExecutionResult.builder()
						.conversationHistory(partialConversationHistory)
						.returnDirect(Objects.requireNonNullElse(returnDirect, false))
						.build();
					throw new ToolCallLimitExceededException(limitBreach.toolName(), limitBreach.limit(),
							partialToolExecutionResult);
				}

				// ToolCallLimitBehavior.RETURN_ERROR_RESPONSE: skip invoking this tool
				// call but keep processing the rest of the batch.
				continue;
			}

			String toolInputArguments = toolCall.arguments();

			// Handle the possible null parameter situation in streaming mode.
			final String finalToolInputArguments;
			if (!StringUtils.hasText(toolInputArguments)) {
				if (logger.isWarnEnabled()) {
					logger.warn("Tool call arguments are null or empty for tool: " + toolName
							+ ". Using empty JSON object as default.");
				}
				finalToolInputArguments = "{}";
			}
			else {
				finalToolInputArguments = toolInputArguments;
			}

			ToolCallback toolCallback = toolCallbacks.stream()
				.filter(tool -> toolName.equals(tool.getToolDefinition().name()))
				.findFirst()
				.orElseGet(() -> this.resolutionFallbackEnabled ? this.toolCallbackResolver.resolve(toolName) : null);

			// 默认只允许当前请求已附带的工具。只有显式启用 fallback，才允许通过 resolver 按名称补查。
			// 因此拥有 resolver 不等于模型能调用其中任意工具，名称必须与定义精确对应。
			if (toolCallback == null) {
				if (logger.isWarnEnabled()) {
					logger.warn(POSSIBLE_LLM_TOOL_NAME_CHANGE_WARNING_START + toolName
							+ POSSIBLE_LLM_TOOL_NAME_CHANGE_WARNING_END);
				}
				throw new IllegalStateException("No ToolCallback found for tool name: " + toolName);
			}

			// 对本批已解析回调的 returnDirect 做 AND：存在一个需要模型继续处理的工具，就继续模型循环。
			// 该标志不会使 for 循环提前结束；本批工具都处理完成后才由 Advisor 决定直接返回。
			if (returnDirect == null) {
				returnDirect = toolCallback.getToolMetadata().returnDirect();
			}
			else {
				returnDirect = returnDirect && toolCallback.getToolMetadata().returnDirect();
			}

			// In streaming/reactive mode the parent observation is propagated through the
			// Reactor context captured in ToolCallReactiveContextHolder. In blocking mode
			// that holder is never populated, so fall back to the observation currently
			// in scope on the calling thread to keep the observation hierarchy intact.
			Observation parent = ToolCallReactiveContextHolder.getContext()
				.getOrDefault(ObservationThreadLocalAccessor.KEY, this.observationRegistry.getCurrentObservation());

			ToolCallingObservationContext observationContext = ToolCallingObservationContext.builder()
				.toolDefinition(toolCallback.getToolDefinition())
				.toolMetadata(toolCallback.getToolMetadata())
				.toolCallId(toolCall.id())
				.toolType(toolCall.type())
				.toolCallArguments(finalToolInputArguments)
				.build();

			String toolCallResult = ToolCallingObservationDocumentation.TOOL_CALL
				.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
						this.observationRegistry)
				.parentObservation(parent)
				.observe(() -> {
					String toolResult;
					try {
						toolResult = toolCallback.call(finalToolInputArguments, toolContext);
					}
					catch (ToolExecutionException ex) {
						// 只捕获此协议异常。默认处理器把 RuntimeException cause 转为结果文本，
						// 但 checked exception、Error、配置为重抛的异常会继续向上失败；未包装的其他异常也会向上抛出。
						toolResult = this.toolExecutionExceptionProcessor.process(ex);
					}
					observationContext.setToolCallResult(toolResult);
					return toolResult;
				});

			// 响应沿用模型生成的调用 ID，使下轮模型能把每个结果对应回原来的工具请求。
			toolResponses.add(new ToolResponseMessage.ToolResponse(toolCall.id(), toolName,
					toolCallResult != null ? toolCallResult : ""));
		}

		return new InternalToolExecutionResult(ToolResponseMessage.builder().responses(toolResponses).build(),
				Objects.requireNonNullElse(returnDirect, false));
	}

	private List<Message> buildConversationHistoryAfterToolExecution(List<Message> previousMessages,
			AssistantMessage assistantMessage, ToolResponseMessage toolResponseMessage) {
		// 必须保留发起工具调用的 assistant 消息，再追加聚合了本批结果的 tool 消息。
		// 只把结果字符串拼到用户消息会丢失角色与 toolCallId 的对应关系；这里始终构造新列表。
		List<Message> messages = new ArrayList<>(previousMessages);
		messages.add(assistantMessage);
		messages.add(toolResponseMessage);
		return messages;
	}

	public void setObservationConvention(ToolCallingObservationConvention observationConvention) {
		this.observationConvention = observationConvention;
	}

	public static Builder builder() {
		return new Builder();
	}

	private record InternalToolExecutionResult(ToolResponseMessage toolResponseMessage, boolean returnDirect) {
	}

	public final static class Builder {

		private ObservationRegistry observationRegistry = DEFAULT_OBSERVATION_REGISTRY;

		private ToolCallbackResolver toolCallbackResolver = DEFAULT_TOOL_CALLBACK_RESOLVER;

		private ToolExecutionExceptionProcessor toolExecutionExceptionProcessor = DEFAULT_TOOL_EXECUTION_EXCEPTION_PROCESSOR;

		private int defaultMaxCallsPerTool = DEFAULT_MAX_CALLS_PER_TOOL;

		private final Map<String, Integer> maxCallsPerTool = new HashMap<>();

		private final Set<String> toolsExcludedFromLimit = new HashSet<>();

		private int maxTotalToolCalls = DEFAULT_MAX_TOTAL_TOOL_CALLS;

		private ToolCallLimitBehavior onLimitExceeded = ToolCallLimitBehavior.THROW;

		private boolean resolutionFallbackEnabled = DEFAULT_RESOLUTION_FALLBACK_ENABLED;

		private Builder() {
		}

		public Builder observationRegistry(ObservationRegistry observationRegistry) {
			this.observationRegistry = observationRegistry;
			return this;
		}

		public Builder toolCallbackResolver(ToolCallbackResolver toolCallbackResolver) {
			this.toolCallbackResolver = toolCallbackResolver;
			return this;
		}

		public Builder toolExecutionExceptionProcessor(
				ToolExecutionExceptionProcessor toolExecutionExceptionProcessor) {
			this.toolExecutionExceptionProcessor = toolExecutionExceptionProcessor;
			return this;
		}

		/**
		 * Default maximum number of times any single tool can be called within a turn,
		 * unless overridden for a specific tool name via
		 * {@link #maxCallsPerTool(String, int)} or exempted via
		 * {@link #excludeToolFromLimit(String)}. Defaults to
		 * {@link DefaultToolCallingManager#DEFAULT_MAX_CALLS_PER_TOOL}; call
		 * {@link #unlimitedCallsPerTool()} to disable this limit entirely.
		 */
		public Builder maxCallsPerTool(int maxCallsPerTool) {
			Assert.isTrue(maxCallsPerTool > 0, "maxCallsPerTool must be greater than 0");
			this.defaultMaxCallsPerTool = maxCallsPerTool;
			return this;
		}

		/**
		 * Disable the per-tool call limit entirely, undoing {@link #maxCallsPerTool(int)}
		 * (or the {@link DefaultToolCallingManager#DEFAULT_MAX_CALLS_PER_TOOL} default),
		 * so that, absent a specific {@link #maxCallsPerTool(String, int)} override, any
		 * tool can be called an unlimited number of times.
		 */
		public Builder unlimitedCallsPerTool() {
			this.defaultMaxCallsPerTool = ToolCallLimits.NO_LIMIT;
			return this;
		}

		/**
		 * Maximum number of times the named tool can be called within a turn, overriding
		 * the default set via {@link #maxCallsPerTool(int)} for that tool only.
		 */
		public Builder maxCallsPerTool(String toolName, int maxCallsPerTool) {
			Assert.hasText(toolName, "toolName cannot be null or empty");
			Assert.isTrue(maxCallsPerTool > 0, "maxCallsPerTool must be greater than 0");
			this.maxCallsPerTool.put(toolName, maxCallsPerTool);
			this.toolsExcludedFromLimit.remove(toolName);
			return this;
		}

		/**
		 * Exempt the named tool from the per-tool call limit, so it can be called an
		 * unlimited number of times regardless of {@link #maxCallsPerTool(int)} or a
		 * prior {@link #maxCallsPerTool(String, int)} override for that tool. Calls to
		 * this tool still count toward {@link #maxTotalToolCalls(int)}.
		 */
		public Builder excludeToolFromLimit(String toolName) {
			Assert.hasText(toolName, "toolName cannot be null or empty");
			this.toolsExcludedFromLimit.add(toolName);
			this.maxCallsPerTool.remove(toolName);
			return this;
		}

		/**
		 * Maximum number of tool calls, across all tools combined, allowed within a turn.
		 * Defaults to {@link DefaultToolCallingManager#DEFAULT_MAX_TOTAL_TOOL_CALLS};
		 * call {@link #unlimitedTotalToolCalls()} to disable this limit entirely.
		 */
		public Builder maxTotalToolCalls(int maxTotalToolCalls) {
			Assert.isTrue(maxTotalToolCalls > 0, "maxTotalToolCalls must be greater than 0");
			this.maxTotalToolCalls = maxTotalToolCalls;
			return this;
		}

		/**
		 * Disable the total tool call limit entirely, undoing
		 * {@link #maxTotalToolCalls(int)} (or the
		 * {@link DefaultToolCallingManager#DEFAULT_MAX_TOTAL_TOOL_CALLS} default).
		 */
		public Builder unlimitedTotalToolCalls() {
			this.maxTotalToolCalls = ToolCallLimits.NO_LIMIT;
			return this;
		}

		/**
		 * What to do when a configured limit is exceeded. Defaults to
		 * {@link ToolCallLimitBehavior#THROW}.
		 */
		public Builder onLimitExceeded(ToolCallLimitBehavior onLimitExceeded) {
			Assert.notNull(onLimitExceeded, "onLimitExceeded cannot be null");
			this.onLimitExceeded = onLimitExceeded;
			return this;
		}

		/**
		 * Whether a call to a tool that is not part of the request's tool callbacks may
		 * still be executed by resolving it through the {@link ToolCallbackResolver}.
		 * Defaults to {@code false}, so that the tools attached to a request bound what
		 * can be executed for that request.
		 * @param resolutionFallbackEnabled {@code true} to allow name-based resolution of
		 * tools absent from the request via the resolver
		 * @since 2.0.1
		 */
		public Builder resolutionFallbackEnabled(boolean resolutionFallbackEnabled) {
			this.resolutionFallbackEnabled = resolutionFallbackEnabled;
			return this;
		}

		public DefaultToolCallingManager build() {
			ToolCallLimits toolCallLimits = new ToolCallLimits(this.defaultMaxCallsPerTool, this.maxCallsPerTool,
					this.toolsExcludedFromLimit, this.maxTotalToolCalls, this.onLimitExceeded);
			return new DefaultToolCallingManager(this.observationRegistry, this.toolCallbackResolver,
					this.toolExecutionExceptionProcessor, toolCallLimits, this.resolutionFallbackEnabled);
		}

	}

}
