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

package org.springframework.ai.chat.client.advisor;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

import org.springframework.ai.chat.client.ChatClientMessageAggregator;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.BaseAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisor;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisorChain;
import org.springframework.ai.chat.client.advisor.api.ToolAdvisor;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallLimitExceededException;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.ToolExecutionEligibilityChecker;
import org.springframework.ai.model.tool.ToolExecutionResult;
import org.springframework.ai.model.tool.internal.ToolCallReactiveContextHolder;
import org.springframework.core.Ordered;
import org.springframework.util.Assert;

/**
 * Recursive Advisor that disables the internal tool execution flow and instead implements
 * the tool calling loop as part of the advisor chain.
 * <p>
 * It uses the CallAdvisorChainUtil to implement looping advisor chain calls.
 * <p>
 * This enables intercepting the tool calling loop by the rest of the advisors next in the
 * chain.
 * <p>
 * When the prompt options do not implement
 * {@link org.springframework.ai.model.tool.ToolCallingChatOptions} (e.g. for models that
 * don't support tool calling), this advisor passes the request through to the next
 * advisor without modification.
 *
 * <p>
 * 中文导读：这是 ChatClient 的工具循环入口。当前主链由 Advisor 发起一轮模型请求，
 * 从 ChatResponse 识别工具调用，交给 ToolCallingManager 执行，再携带工具结果发起下一轮。
 * 模型负责生成工具名称和参数，实际 Java 方法或 MCP 请求由应用中的 ToolCallback 执行。
 *
 * <p>
 * 每轮使用 {@code chain.copy(this)} 重建本 Advisor 之后的链，
 * 因而下游 Advisor 可以观察每一轮模型请求，位于本 Advisor 之前的外层 Advisor 只包围整个循环。
 * 默认记忆 Advisor 的 order 更小，位于外层；本类用局部消息历史保留工具的中间请求与结果。
 *
 * <p>
 * 终止条件包括模型不再请求工具、工具批次要求 returnDirect，以及 Manager 以异常报告工具调用超限。
 * 同步路径用 do/while，流式路径则等当前响应正常结束并聚合完毕后再衔接下一段 Flux。
 * 每轮模型的 token 用量由 UsageAccumulator 累计，避免最终结果只报告最后一次模型调用的用量。
 *
 * @author Christian Tzolov
 * @author Jewoo Shin
 * @since 2.0.0
 */
public class ToolCallingAdvisor implements CallAdvisor, StreamAdvisor, ToolAdvisor {

	private static final ChatClientMessageAggregator CHAT_CLIENT_MESSAGE_AGGREGATOR = new ChatClientMessageAggregator();

	/**
	 * Default advisor order. Higher than
	 * {@link org.springframework.ai.chat.client.advisor.api.Advisor#DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER}
	 * (+200) so that memory advisors at their default order are placed outside this
	 * advisor and do not participate in each tool-call iteration.
	 * <p>
	 * order 数值较小的 Advisor 先处理请求、最后处理响应；默认记忆 +200 在工具 +300 的外层。
	 */
	public static final int DEFAULT_ORDER = Ordered.HIGHEST_PRECEDENCE + 300;

	protected static final ToolExecutionEligibilityChecker DEFAULT_TOOL_EXECUTION_ELIGIBILITY_CHECKER = chatResponse -> chatResponse != null
			&& chatResponse.hasToolCalls();

	protected final ToolCallingManager toolCallingManager;

	private final ToolExecutionEligibilityChecker toolExecutionEligibilityChecker;

	/**
	 * Set the order close to {@link Ordered#LOWEST_PRECEDENCE} to ensure an advisor is
	 * executed first in the chain (first for request processing, last for response
	 * processing).
	 * <p>
	 * https://docs.spring.io/spring-ai/reference/api/advisors.html#_advisor_order
	 */
	private final int advisorOrder;

	private final boolean conversationHistoryEnabled;

	protected ToolCallingAdvisor(ToolCallingManager toolCallingManager,
			ToolExecutionEligibilityChecker toolExecutionEligibilityChecker, int advisorOrder,
			boolean conversationHistoryEnabled) {
		Assert.notNull(toolCallingManager, "toolCallingManager must not be null");
		Assert.notNull(toolExecutionEligibilityChecker, "toolExecutionEligibilityChecker must not be null");
		Assert.isTrue(advisorOrder > BaseAdvisor.HIGHEST_PRECEDENCE && advisorOrder < BaseAdvisor.LOWEST_PRECEDENCE,
				"advisorOrder must be between HIGHEST_PRECEDENCE and LOWEST_PRECEDENCE");

		this.toolCallingManager = toolCallingManager;
		this.toolExecutionEligibilityChecker = toolExecutionEligibilityChecker;
		this.advisorOrder = advisorOrder;
		this.conversationHistoryEnabled = conversationHistoryEnabled;
	}

	@Override
	public String getName() {
		return "Tool Calling Advisor";
	}

	@Override
	public int getOrder() {
		return this.advisorOrder;
	}

	// -------------------------------------------------------------------------
	// Call (non-streaming) implementation
	// -------------------------------------------------------------------------
	@Override
	public ChatClientResponse adviseCall(ChatClientRequest chatClientRequest, CallAdvisorChain callAdvisorChain) {
		Assert.notNull(callAdvisorChain, "callAdvisorChain must not be null");
		Assert.notNull(chatClientRequest, "chatClientRequest must not be null");

		ChatOptions options = chatClientRequest.prompt().getOptions();
		// 选项随 ChatClientRequest 中的 Prompt 传入；不支持工具选项的模型直接走普通 Advisor 链。
		if (!(options instanceof ToolCallingChatOptions toolCallingChatOptions)) {
			// No tool calling options - skip the tool calling advisor
			return callAdvisorChain.nextCall(chatClientRequest);
		}

		chatClientRequest = this.doInitializeLoop(chatClientRequest, callAdvisorChain);

		var instructions = chatClientRequest.prompt().getInstructions();
		// Always the complete history for this turn, independent of what
		// doGetNextInstructionsForToolCall trims `instructions` down to for forwarding
		// to the rest of the chain (e.g. for a memory advisor nested inside the loop
		// that doesn't itself de-duplicate). This is what the ToolCallingManager sees,
		// so trimming the forwarded prompt never hides earlier rounds from its tool
		// call limit counting.
		var fullTurnHistory = instructions;

		// instructions 用于向下游转发；fullTurnHistory 用于完整记录本轮工具结果和限额统计。
		// 两者在关闭内部历史时会分开，不能用裁剪后的转发列表代替 Manager 的完整输入。
		ChatClientResponse chatClientResponse = null;
		UsageAccumulator usageAccumulator = new UsageAccumulator();

		boolean isToolCall = false;

		do {

			// Before Call
			var processedChatClientRequest = ChatClientRequest.builder()
				.prompt(new Prompt(instructions, toolCallingChatOptions))
				.context(chatClientRequest.context())
				.build();

			// Next Call
			// 扩展 hook 可以按轮次动态选择工具；下方会重新读取它处理后的选项用于本轮执行。
			processedChatClientRequest = this.doBeforeCall(processedChatClientRequest, callAdvisorChain);

			// Re-read the options doBeforeCall() may have mutated (e.g. a subclass
			// injecting per-iteration tool callbacks) so executeToolCalls() below sees
			// what was actually sent to the model, not the pre-doBeforeCall snapshot.
			toolCallingChatOptions = (ToolCallingChatOptions) processedChatClientRequest.prompt().getOptions();
			Assert.notNull(toolCallingChatOptions,
					"redundant check that should never fail (doBeforeCall must not change the options type away from ToolCallingChatOptions), but here to help NullAway");

			// Advisor 链的游标会随 nextCall 消耗；每轮复制“本节点之后”的链，才能再次到达模型节点。
			// 复制不包含当前工具 Advisor，避免把每一轮又包装为一个新的完整工具循环。
			chatClientResponse = callAdvisorChain.copy(this).nextCall(processedChatClientRequest);

			chatClientResponse = this.doAfterCall(chatClientResponse, callAdvisorChain);

			// After Call
			ChatResponse chatResponse = chatClientResponse.chatResponse();
			usageAccumulator.addRoundResponse(chatResponse);
			isToolCall = this.toolExecutionEligibilityChecker.isToolCallResponse(chatResponse);

			if (isToolCall) {
				Assert.notNull(chatResponse, "redundant check that should never fail, but here to help NullAway");

				ToolExecutionResult toolExecutionResult;
				try {
					// 同步路径就在当前调用线程执行工具；这里没有异步派发或工具并行化。
					toolExecutionResult = this.toolCallingManager
						.executeToolCalls(new Prompt(fullTurnHistory, toolCallingChatOptions), chatResponse);
				}
				catch (ToolCallLimitExceededException ex) {
					// 限额是本层显式处理的终止信号；其他未被 Manager 转为工具结果的异常继续向上抛出。
					// Return the breach as a single generation - directly to the
					// application client instead of looping back to the LLM.
					chatClientResponse = chatClientResponse.mutate()
						.chatResponse(ChatResponse.builder()
							.from(chatResponse)
							.generations(List.of(ex.buildGeneration()))
							.build())
						.build();
					break;
				}

				fullTurnHistory = toolExecutionResult.conversationHistory();

				if (toolExecutionResult.returnDirect()) {

					// 把本批 tool 结果转换为 AssistantMessage/Generation，保留工具 ID、名称和结束原因。
					// 直接返回意味着无需再请模型解释结果，并不表示之前的模型请求或工具执行被跳过。
					// Return tool execution result directly to the application client
					// instead of returning it to the LLM.
					chatClientResponse = chatClientResponse.mutate()
						.chatResponse(ChatResponse.builder()
							.from(chatResponse)
							.generations(ToolExecutionResult.buildGenerations(toolExecutionResult))
							.build())
						.build();
					break;
				}

				instructions = this.doGetNextInstructionsForToolCall(processedChatClientRequest, chatClientResponse,
						toolExecutionResult);
			}

		}
		while (isToolCall); // loop until no tool calls are present

		chatClientResponse = usageAccumulator.applyAccumulatedUsage(chatClientResponse);
		return this.doFinalizeLoop(chatClientResponse, callAdvisorChain);
	}

	protected List<Message> doGetNextInstructionsForToolCall(ChatClientRequest chatClientRequest,
			ChatClientResponse chatClientResponse, ToolExecutionResult toolExecutionResult) {

		if (!this.conversationHistoryEnabled) {
			// 下游记忆 Advisor 负责补全历史时，只传系统指令和最新 tool response，避免重复追加旧消息。
			// 这一模式需要相应的记忆配置；仅关闭本开关并不会自动建立可续接的完整对话。
			List<Message> history = toolExecutionResult.conversationHistory();
			if (history.isEmpty()) {
				return history;
			}
			return List.of(chatClientRequest.prompt().getSystemMessage(), history.get(history.size() - 1));
		}

		return toolExecutionResult.conversationHistory();
	}

	protected ChatClientResponse doFinalizeLoop(ChatClientResponse chatClientResponse,
			CallAdvisorChain callAdvisorChain) {
		return chatClientResponse;
	}

	protected ChatClientRequest doInitializeLoop(ChatClientRequest chatClientRequest,
			CallAdvisorChain callAdvisorChain) {
		return chatClientRequest;
	}

	protected ChatClientRequest doBeforeCall(ChatClientRequest chatClientRequest, CallAdvisorChain callAdvisorChain) {
		return chatClientRequest;
	}

	protected ChatClientResponse doAfterCall(ChatClientResponse chatClientResponse, CallAdvisorChain callAdvisorChain) {
		return chatClientResponse;
	}

	// -------------------------------------------------------------------------
	// Streaming implementation
	// -------------------------------------------------------------------------
	@Override
	public Flux<ChatClientResponse> adviseStream(ChatClientRequest chatClientRequest,
			StreamAdvisorChain streamAdvisorChain) {
		Assert.notNull(streamAdvisorChain, "streamAdvisorChain must not be null");
		Assert.notNull(chatClientRequest, "chatClientRequest must not be null");

		ChatOptions options = chatClientRequest.prompt().getOptions();
		if (!(options instanceof ToolCallingChatOptions toolCallingChatOptions)) {
			// No tool calling options - skip the tool calling advisor
			return streamAdvisorChain.nextStream(chatClientRequest);
		}

		return Flux.defer(() -> {
			// 初始化放在订阅时，确保重复订阅拥有独立的用量累积状态；构建 Flux 本身不会执行模型或工具。
			ChatClientRequest initializedRequest = this.doInitializeLoopStream(chatClientRequest, streamAdvisorChain);
			// Subscription-local accumulator so usage is not shared across subscriptions.
			UsageAccumulator usageAccumulator = new UsageAccumulator();
			List<Message> initialInstructions = initializedRequest.prompt().getInstructions();
			return this.internalStream(streamAdvisorChain, initializedRequest, toolCallingChatOptions,
					initialInstructions, initialInstructions, usageAccumulator);
		});
	}

	private Flux<ChatClientResponse> internalStream(StreamAdvisorChain streamAdvisorChain,
			ChatClientRequest originalRequest, ToolCallingChatOptions toolCallingChatOptions,
			List<Message> instructions, List<Message> fullTurnHistory, UsageAccumulator usageAccumulator) {

		return Flux.deferContextual(contextView -> {
			// Build request with current instructions
			var processedRequest = ChatClientRequest.builder()
				.prompt(new Prompt(instructions, toolCallingChatOptions))
				.context(originalRequest.context())
				.build();

			processedRequest = this.doBeforeStream(processedRequest, streamAdvisorChain);

			// Get a copy of the chain excluding this advisor
			StreamAdvisorChain chainCopy = streamAdvisorChain.copy(this);

			final ChatClientRequest finalRequest = processedRequest;

			Flux<ChatClientResponse> responseFlux = chainCopy.nextStream(processedRequest);

			// Re-read the options doBeforeStream() may have mutated (e.g. a subclass
			// injecting per-iteration tool callbacks) so executeToolCalls() below sees
			// what was actually sent to the model, not the pre-doBeforeStream snapshot.
			ToolCallingChatOptions processedOptions = (ToolCallingChatOptions) finalRequest.prompt().getOptions();
			Assert.notNull(processedOptions,
					"redundant check that should never fail (doBeforeStream must not change the options type away from ToolCallingChatOptions), but here to help NullAway");

			return streamWithToolCallResponses(responseFlux, finalRequest, fullTurnHistory, streamAdvisorChain,
					originalRequest, processedOptions, usageAccumulator);
		});
	}

	private Flux<ChatClientResponse> streamWithToolCallResponses(Flux<ChatClientResponse> responseFlux,
			ChatClientRequest finalRequest, List<Message> fullTurnHistory, StreamAdvisorChain streamAdvisorChain,
			ChatClientRequest originalRequest, ToolCallingChatOptions optionsCopy, UsageAccumulator usageAccumulator) {

		AtomicReference<ChatClientResponse> aggregatedResponseRef = new AtomicReference<>();
		// Snapshot of the usage accumulated from previous rounds (before this round).
		ChatResponse previousAccumulatedResponse = usageAccumulator.accumulatedResponse();

		return CHAT_CLIENT_MESSAGE_AGGREGATOR.aggregateChatClientResponse(responseFlux, aggregatedResponseRef::set)
			.map(chatClientResponse -> UsageAccumulator.applyPreviousAccumulatedUsageToChunk(chatClientResponse,
					previousAccumulatedResponse))
			.concatWith(Flux.defer(() -> this.handleToolCallRecursion(aggregatedResponseRef.get(), finalRequest,
					fullTurnHistory, streamAdvisorChain, originalRequest, optionsCopy, usageAccumulator)))
			.filter(ccr -> !this.toolExecutionEligibilityChecker.isToolCallResponse(ccr.chatResponse()));
	}

	/**
	 * Handles tool call detection and recursion after streaming completes. Returns empty
	 * flux if no tool call, or recursive stream if tool call detected.
	 * <p>
	 * 聚合器在正常完成时填入完整响应，concatWith 随后进入本方法。
	 * 这样执行工具时可以拿到完整参数，而不是尚未拼完的 JSON 片段；带工具调用的响应块会被外层过滤。
	 * 未携带工具调用的文本块仍可向外传递，因此本类不是缓存所有中间输出后只发布最终回答。
	 * 如果本轮流报错或订阅在完成前取消，则不会启动 concatWith 中的下一轮工具处理。
	 */
	private Flux<ChatClientResponse> handleToolCallRecursion(ChatClientResponse aggregatedResponse,
			ChatClientRequest finalRequest, List<Message> fullTurnHistory, StreamAdvisorChain streamAdvisorChain,
			ChatClientRequest originalRequest, ToolCallingChatOptions optionsCopy, UsageAccumulator usageAccumulator) {

		if (aggregatedResponse == null) {
			return Flux.empty();
		}

		aggregatedResponse = this.doAfterStream(aggregatedResponse, streamAdvisorChain);

		ChatResponse chatResponse = aggregatedResponse.chatResponse();
		ChatResponse accumulatedChatResponse = usageAccumulator.addRoundResponse(chatResponse);
		boolean isToolCall = this.toolExecutionEligibilityChecker.isToolCallResponse(chatResponse);

		if (!isToolCall) {
			// No tool call: the final round's chunks have already been streamed. If that
			// round carried no usage of its own, the usage accumulated from previous
			// rounds was never stamped onto any emitted chunk, so emit a trailing
			// usage-only response to keep the aggregated stream usage correct. Otherwise
			// nothing more needs to be emitted.
			Flux<ChatClientResponse> finalEmissions = UsageAccumulator
				.emitFinalUsageCorrectionIfNecessary(aggregatedResponse, chatResponse, accumulatedChatResponse);
			return this.doFinalizeLoopStream(finalEmissions, streamAdvisorChain);
		}

		Assert.notNull(chatResponse, "redundant check that should never fail, but here to help NullAway");
		final ChatClientResponse finalAggregatedResponse = aggregatedResponse;

		// Execute tool calls on bounded elastic scheduler (tool execution is blocking)
		// ToolCallback 是阻塞接口，所以将本段订阅移到 boundedElastic，避免在模型网络事件线程直接执行。
		// 取消订阅不等于回滚已经执行的工具；正在运行的工具能否停止仍取决于其对中断/取消的支持。
		Flux<ChatClientResponse> toolCallFlux = Flux.deferContextual(ctx -> {
			ToolExecutionResult toolExecutionResult;
			try {
				// 将本订阅 Reactor Context 暂存到当前线程，Manager 用它关联 observation，
				// AsyncMcpToolCallback 用它给内部 block() 创建的新订阅恢复上下文。
				ToolCallReactiveContextHolder.setContext(ctx);
				toolExecutionResult = this.toolCallingManager.executeToolCalls(new Prompt(fullTurnHistory, optionsCopy),
						chatResponse);
			}
			catch (ToolCallLimitExceededException ex) {
				// A configured tool call limit was hit. Return the breach as a single
				// generation - rather than mixing it with any successful calls from
				// the same batch as if they were equally valid alternative answers -
				// directly to the application client instead of looping back to the
				// LLM.
				ChatClientResponse limitResponse = finalAggregatedResponse.mutate()
					.chatResponse(ChatResponse.builder()
						.from(chatResponse)
						.generations(List.of(ex.buildGeneration()))
						.build())
					.build();
				return Flux.just(usageAccumulator.applyAccumulatedUsage(limitResponse));
			}
			finally {
				// boundedElastic 线程会复用，必须清理 ThreadLocal，避免下一次工具调用串入前一次上下文。
				ToolCallReactiveContextHolder.clearContext();
			}

			if (toolExecutionResult.returnDirect()) {
				// Return tool execution result directly to the application client
				ChatClientResponse directResponse = finalAggregatedResponse.mutate()
					.chatResponse(ChatResponse.builder()
						.from(chatResponse)
						.generations(ToolExecutionResult.buildGenerations(toolExecutionResult))
						.build())
					.build();
				return Flux.just(usageAccumulator.applyAccumulatedUsage(directResponse));
			}
			else {
				// Recursive call with updated conversation history
				// 下一段 Flux 在本批工具完成后才建立；完整历史继续传给 Manager，转发历史可由 hook 裁剪。
				List<Message> nextInstructions = this.doGetNextInstructionsForToolCallStream(finalRequest,
						finalAggregatedResponse, toolExecutionResult);
				return this.internalStream(streamAdvisorChain, originalRequest, optionsCopy, nextInstructions,
						toolExecutionResult.conversationHistory(), usageAccumulator);
			}
		});
		return toolCallFlux.subscribeOn(Schedulers.boundedElastic());
	}

	/**
	 * Hook method called at the start of the streaming tool call loop. Subclasses can
	 * override to customize initialization behavior.
	 * @param chatClientRequest the initial request
	 * @param streamAdvisorChain the stream advisor chain
	 * @return the potentially modified request
	 */
	protected ChatClientRequest doInitializeLoopStream(ChatClientRequest chatClientRequest,
			StreamAdvisorChain streamAdvisorChain) {
		return chatClientRequest;
	}

	/**
	 * Hook method called before each streaming call in the tool call loop. Subclasses can
	 * override to customize pre-call behavior.
	 * @param chatClientRequest the request about to be processed
	 * @param streamAdvisorChain the stream advisor chain
	 * @return the potentially modified request
	 */
	protected ChatClientRequest doBeforeStream(ChatClientRequest chatClientRequest,
			StreamAdvisorChain streamAdvisorChain) {
		return chatClientRequest;
	}

	/**
	 * Hook method called after each streaming call in the tool call loop. Subclasses can
	 * override to customize post-call behavior.
	 * @param chatClientResponse the response from the call
	 * @param streamAdvisorChain the stream advisor chain
	 * @return the potentially modified response
	 */
	protected ChatClientResponse doAfterStream(ChatClientResponse chatClientResponse,
			StreamAdvisorChain streamAdvisorChain) {
		return chatClientResponse;
	}

	/**
	 * Hook method called at the end of the streaming tool call loop to finalize the
	 * response. Subclasses can override to customize finalization behavior.
	 * @param chatClientResponseFlux the flux of collected response chunks to emit
	 * @param streamAdvisorChain the stream advisor chain
	 * @return the potentially modified flux of responses
	 */
	protected Flux<ChatClientResponse> doFinalizeLoopStream(Flux<ChatClientResponse> chatClientResponseFlux,
			StreamAdvisorChain streamAdvisorChain) {
		// 当前调用点位于“本轮没有工具调用”的结束分支，returnDirect 和限额分支会直接返回自己的 Flux。
		return chatClientResponseFlux;
	}

	/**
	 * Hook method to determine the next instructions for a tool call iteration in
	 * streaming mode. Subclasses can override to customize conversation history handling.
	 * @param chatClientRequest the current request
	 * @param chatClientResponse the current response
	 * @param toolExecutionResult the result of tool execution
	 * @return the list of messages to use as instructions for the next iteration
	 */
	protected List<Message> doGetNextInstructionsForToolCallStream(ChatClientRequest chatClientRequest,
			ChatClientResponse chatClientResponse, ToolExecutionResult toolExecutionResult) {

		// 与同步路径相同：是否携带完整历史由工具循环与下游记忆 Advisor 的职责分配决定。
		if (!this.conversationHistoryEnabled) {
			List<Message> history = toolExecutionResult.conversationHistory();
			if (history.isEmpty()) {
				return history;
			}
			return List.of(chatClientRequest.prompt().getSystemMessage(), history.get(history.size() - 1));
		}

		return toolExecutionResult.conversationHistory();
	}

	/**
	 * Creates a new Builder instance for constructing a ToolCallingAdvisor.
	 * @return a new Builder instance
	 */
	public static Builder<?> builder() {
		return new Builder<>();
	}

	/**
	 * Builder for creating instances of ToolCallingAdvisor.
	 * <p>
	 * This builder uses the self-referential generic pattern to support extensibility.
	 *
	 * @param <T> the builder type, used for self-referential generics to support method
	 * chaining in subclasses
	 */
	public static class Builder<T extends Builder<T>> {

		private ToolCallingManager toolCallingManager = ToolCallingManager.builder().build();

		private ToolExecutionEligibilityChecker toolExecutionEligibilityChecker = DEFAULT_TOOL_EXECUTION_ELIGIBILITY_CHECKER;

		private int advisorOrder = DEFAULT_ORDER;

		private boolean conversationHistoryEnabled = true;

		protected Builder() {
		}

		/**
		 * Returns this builder cast to the appropriate type for method chaining.
		 * Subclasses should override this method to return the correct type.
		 * @return this builder instance
		 */
		@SuppressWarnings("unchecked")
		protected T self() {
			return (T) this;
		}

		/**
		 * Sets the ToolCallingManager to be used by the advisor.
		 * @param toolCallingManager the ToolCallingManager instance
		 * @return this Builder instance for method chaining
		 */
		public T toolCallingManager(ToolCallingManager toolCallingManager) {
			this.toolCallingManager = toolCallingManager;
			return self();
		}

		/**
		 * Sets the checker that determines whether a model response should trigger tool
		 * execution. Defaults to
		 * {@code chatResponse -> chatResponse != null && chatResponse.hasToolCalls()}.
		 * Override to apply provider-specific stop-reason logic.
		 * @param toolExecutionEligibilityChecker the checker
		 * @return this Builder instance for method chaining
		 */
		public T toolExecutionEligibilityChecker(ToolExecutionEligibilityChecker toolExecutionEligibilityChecker) {
			this.toolExecutionEligibilityChecker = toolExecutionEligibilityChecker;
			return self();
		}

		/**
		 * Sets the order of the advisor in the advisor chain.
		 * @param advisorOrder the order value, must be between HIGHEST_PRECEDENCE and
		 * LOWEST_PRECEDENCE
		 * @return this Builder instance for method chaining
		 */
		public T advisorOrder(int advisorOrder) {
			this.advisorOrder = advisorOrder;
			return self();
		}

		/**
		 * Sets whether internal conversation history is enabled. If false, you need a
		 * ChatMemory Advisor registered next in the chain.
		 * @param conversationHistoryEnabled true to enable, false to disable
		 * @return this Builder instance for method chaining
		 */
		public T conversationHistoryEnabled(boolean conversationHistoryEnabled) {
			this.conversationHistoryEnabled = conversationHistoryEnabled;
			return self();
		}

		/**
		 * Disables internal conversation history. You need a ChatMemory Advisor
		 * registered next in the chain.
		 * @return this Builder instance for method chaining
		 */
		public T disableInternalConversationHistory() {
			this.conversationHistoryEnabled = false;
			return self();
		}

		/**
		 * Returns the configured ToolCallingManager.
		 * @return the ToolCallingManager instance
		 */
		protected ToolCallingManager getToolCallingManager() {
			return this.toolCallingManager;
		}

		/**
		 * Returns the configured advisor order.
		 * @return the advisor order value
		 */
		public int getAdvisorOrder() {
			return this.advisorOrder;
		}

		/**
		 * Returns the configured ToolExecutionEligibilityChecker.
		 * @return the ToolExecutionEligibilityChecker instance
		 */
		public ToolExecutionEligibilityChecker getToolExecutionEligibilityChecker() {
			return this.toolExecutionEligibilityChecker;
		}

		/**
		 * Creates a shallow copy of this builder with all current settings. The copy can
		 * be used as a template from which per-call overrides (e.g.
		 * {@code conversationHistoryEnabled}) are applied without mutating the original.
		 * <p>
		 * Subclasses must override {@link #newCopy()} to return an instance of their own
		 * type, and override this method to copy their additional fields into it.
		 * @return a new {@link Builder} with the same configuration
		 */
		public Builder<?> copy() {
			Builder<?> copy = newCopy();
			copy.toolCallingManager = this.toolCallingManager;
			copy.toolExecutionEligibilityChecker = this.toolExecutionEligibilityChecker;
			copy.advisorOrder = this.advisorOrder;
			copy.conversationHistoryEnabled = this.conversationHistoryEnabled;
			return copy;
		}

		/**
		 * Factory method called by {@link #copy()} to create the blank builder instance
		 * that will receive the copied fields. Subclasses must override this to return an
		 * instance of their own builder type so that {@link #copy()} preserves the full
		 * subtype.
		 * @return a new, unconfigured builder of the same concrete type
		 */
		protected Builder<?> newCopy() {
			return new Builder<>();
		}

		/**
		 * Returns whether internal conversation history is enabled.
		 * @return true if internal conversation history is enabled, false if disabled
		 */
		protected boolean isConversationHistoryEnabled() {
			return this.conversationHistoryEnabled;
		}

		/**
		 * Builds and returns a new ToolCallingAdvisor instance with the configured
		 * properties.
		 * @return a new ToolCallingAdvisor instance
		 * @throws IllegalArgumentException if toolCallingManager is null or advisorOrder
		 * is out of valid range
		 */
		public ToolCallingAdvisor build() {
			return new ToolCallingAdvisor(this.toolCallingManager, this.toolExecutionEligibilityChecker,
					this.advisorOrder, this.conversationHistoryEnabled);
		}

	}

}
