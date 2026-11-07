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

package org.springframework.ai.openai;

import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.openai.client.OpenAIClient;
import com.openai.client.OpenAIClientAsync;
import com.openai.core.JsonValue;
import com.openai.core.RequestOptions;
import com.openai.core.http.AsyncStreamResponse;
import com.openai.core.http.HttpResponseFor;
import com.openai.errors.OpenAIInvalidDataException;
import com.openai.models.FunctionDefinition;
import com.openai.models.FunctionParameters;
import com.openai.models.ReasoningEffort;
import com.openai.models.ResponseFormatJsonObject;
import com.openai.models.ResponseFormatJsonSchema;
import com.openai.models.ResponseFormatText;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionAssistantMessageParam;
import com.openai.models.chat.completions.ChatCompletionChunk;
import com.openai.models.chat.completions.ChatCompletionChunk.Choice;
import com.openai.models.chat.completions.ChatCompletionChunk.Choice.Delta;
import com.openai.models.chat.completions.ChatCompletionChunk.Choice.Delta.ToolCall;
import com.openai.models.chat.completions.ChatCompletionChunk.Choice.Delta.ToolCall.Function;
import com.openai.models.chat.completions.ChatCompletionChunk.Choice.FinishReason;
import com.openai.models.chat.completions.ChatCompletionContentPart;
import com.openai.models.chat.completions.ChatCompletionContentPartImage;
import com.openai.models.chat.completions.ChatCompletionContentPartInputAudio;
import com.openai.models.chat.completions.ChatCompletionContentPartText;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.chat.completions.ChatCompletionFunctionTool;
import com.openai.models.chat.completions.ChatCompletionMessage;
import com.openai.models.chat.completions.ChatCompletionMessageFunctionToolCall;
import com.openai.models.chat.completions.ChatCompletionMessageParam;
import com.openai.models.chat.completions.ChatCompletionMessageToolCall;
import com.openai.models.chat.completions.ChatCompletionNamedToolChoice;
import com.openai.models.chat.completions.ChatCompletionStreamOptions;
import com.openai.models.chat.completions.ChatCompletionTool;
import com.openai.models.chat.completions.ChatCompletionToolChoiceOption;
import com.openai.models.chat.completions.ChatCompletionToolMessageParam;
import com.openai.models.chat.completions.ChatCompletionUserMessageParam;
import com.openai.models.completions.CompletionUsage;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationRegistry;
import io.micrometer.observation.contextpropagation.ObservationThreadLocalAccessor;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jspecify.annotations.Nullable;
import reactor.core.publisher.Flux;
import tools.jackson.databind.JsonNode;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.chat.metadata.EmptyRateLimit;
import org.springframework.ai.chat.metadata.EmptyUsage;
import org.springframework.ai.chat.metadata.RateLimit;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.MessageAggregator;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.observation.ChatModelObservationConvention;
import org.springframework.ai.chat.observation.ChatModelObservationDocumentation;
import org.springframework.ai.chat.observation.DefaultChatModelObservationConvention;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.content.Media;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.observation.conventions.AiProvider;
import org.springframework.ai.openai.http.okhttp.OpenAiHttpClientBuilderCustomizer;
import org.springframework.ai.openai.metadata.OpenAiRateLimit;
import org.springframework.ai.openai.setup.OpenAiSetup;
import org.springframework.ai.support.UsageCalculator;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.ai.util.JacksonUtils;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.MimeTypeUtils;
import org.springframework.util.StringUtils;

/**
 * Chat Model implementation using the OpenAI Java SDK.
 * <p>
 * 中文导读：这是 Spring AI 的 {@link ChatModel} 到 OpenAI Chat Completions API 的适配层。
 * 输入是包含消息和选项的 {@link Prompt}，输出是包含候选回答与用量的 {@link ChatResponse}；
 * HTTP 发送、协议反序列化和客户端重试交给 OpenAI Java SDK。
 * <p>
 * 同步入口按 {@code call -> buildRequestPrompt -> internalCall -> createRequest} 阅读；
 * 流式入口按 {@code stream -> internalStream -> ChunkMerger -> buildGeneration} 阅读。
 * 两条路径都把工具定义发送给模型，并把模型请求的工具调用放进 AssistantMessage，
 * 本类不执行本地工具或发起工具结果回传后的下一轮请求；多轮编排位于 {@code ToolCallingAdvisor}。
 * <p>
 * 本类只在 Prompt 没有 options 时采用模型默认选项；已有 options 会整体沿用，
 * {@code createRequest} 需要它们是 {@link OpenAiChatOptions}。
 * 使用 ChatClient 时，通用选项与模型选项的组合还要向上阅读客户端构造 Prompt 的过程。
 *
 * @author Julien Dubois
 * @author Christian Tzolov
 * @author Soby Chacko
 * @author Ilayaperumal Gopinathan
 * @author Thomas Vitale
 * @author Eric Bottard
 * @author Taewoong Kim
 * @author Jewoo Shin
 * @author guan xu
 * @author Raphael Vullriede
 */
public final class OpenAiChatModel implements ChatModel {

	private static final ChatModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultChatModelObservationConvention();

	private static final String REASONING_CONTENT = "reasoningContent";

	static final String TOOL_CALL_ADDITIONAL_PROPERTIES_METADATA_KEY = "openai.tool_calls.additional_properties";

	private static final TypeReference<Map<String, Object>> MAP_TYPE_REF = new TypeReference<>() {
	};

	// Jackson 2 required due to OpenAI deserializers
	private static final ObjectMapper objectMapper = new ObjectMapper();

	private final Log logger = LogFactory.getLog(OpenAiChatModel.class);

	private final OpenAIClient openAiClient;

	private final OpenAIClientAsync openAiClientAsync;

	private final OpenAiChatOptions options;

	private final ObservationRegistry observationRegistry;

	private final ToolCallingManager toolCallingManager;

	private ChatModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	/**
	 * Creates a new builder for {@link OpenAiChatModel}.
	 * @return a new builder instance
	 */
	public static Builder builder() {
		return new Builder();
	}

	private OpenAiChatModel(OpenAIClient openAiClient, OpenAIClientAsync openAiClientAsync, OpenAiChatOptions options,
			ObservationRegistry observationRegistry, ToolCallingManager toolCallingManager) {
		this.openAiClient = openAiClient;
		this.openAiClientAsync = openAiClientAsync;
		this.options = options;
		this.observationRegistry = observationRegistry;
		this.toolCallingManager = toolCallingManager;
	}

	/**
	 * Gets the chat options for this model.
	 * @return the chat options
	 * @since 2.0.0
	 */
	@Override
	public OpenAiChatOptions getOptions() {
		return this.options;
	}

	/**
	 * @deprecated use {@link #getOptions()} instead.
	 */
	@Override
	@Deprecated(forRemoval = true)
	@SuppressWarnings("removal")
	public ChatOptions getDefaultOptions() {
		return this.options;
	}

	@Override
	public ChatResponse call(Prompt prompt) {
		// 先补齐缺失的整份 options，再校验已知不支持的选项；这里没有逐字段默认值合并。
		Prompt requestPrompt = buildRequestPrompt(prompt);
		verifyPromptChatOptions(requestPrompt);
		return this.internalCall(requestPrompt, null);
	}

	/**
	 * Internal method to handle chat completion calls with tool execution support.
	 * <p>
	 * 当前实现只执行一轮远程模型调用。公开入口传入的 previousChatResponse 为 null，
	 * 保留的用量累加参数不代表这里仍有本地工具执行循环。
	 * @param prompt the prompt for the chat completion
	 * @param previousChatResponse the previous chat response for accumulating usage
	 * @return the chat response
	 */
	private ChatResponse internalCall(Prompt prompt, @Nullable ChatResponse previousChatResponse) {

		// 1. 分别构建 API 请求体和 SDK 的请求级选项。消息/Schema 转换失败发生在 Observation 之前。
		ChatCompletionCreateParams request = this.createRequest(prompt, false);
		RequestOptions requestOptions = this.buildRequestOptions(prompt);

		ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
			.prompt(prompt)
			.provider(AiProvider.OPENAI.value())
			.build();

		// 2. observe 在调用线程中包住一次模型操作，并负责记录异常、关闭 Scope 和停止 Observation。
		// 父 Observation 来自当前线程的作用域；OpenAiSetup 创建的客户端还会独立记录 HTTP 尝试。
		// 本层没有 RetryTemplate/retry 操作符；SDK 使用客户端构造时确定的重试配置。
		ChatResponse response = ChatModelObservationDocumentation.CHAT_MODEL_OPERATION
			.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> {

				// The raw response carries the x-ratelimit-* headers alongside the parsed
				// body; the plain create(...) call discards them.
				HttpResponseFor<ChatCompletion> rawResponse = this.openAiClient.chat()
					.completions()
					.withRawResponse()
					.create(request, requestOptions);
				ChatCompletion chatCompletion = rawResponse.parse();
				RateLimit rateLimit = OpenAiRateLimit.from(rawResponse.headers());

				// 3. 一个 choice 对应一个 Generation；空候选按现有契约返回空结果列表。
				List<ChatCompletion.Choice> choices = chatCompletion.choices();
				if (choices.isEmpty()) {
					if (logger.isWarnEnabled()) {
						logger.warn("No choices returned for prompt: " + prompt);
					}
					return new ChatResponse(List.of());
				}

				List<Generation> generations = choices.stream().map(choice -> {
					Map<String, Object> metadata = Map.of("id", chatCompletion.id(), "role",
							choice.message()._role().asString().isPresent() ? choice.message()._role().asStringOrThrow()
									: "",
							"index", choice.index(), "finishReason", choice.finishReason().value().toString(),
							"refusal", choice.message().refusal().orElse(""), "annotations",
							choice.message().annotations().orElse((List) List.of(Map.of())), REASONING_CONTENT,
							getReasoningContent(choice));
					return buildGeneration(choice, metadata, request);
				}).toList();

				// 4. 生成级信息放在 Generation，整次响应的模型名、用量和限流头放在 ChatResponseMetadata。
				// Current usage
				CompletionUsage usage = chatCompletion.usage().orElse(null);
				Usage currentChatResponseUsage = usage != null ? getDefaultUsage(usage) : new EmptyUsage();
				Usage accumulatedUsage = UsageCalculator.getCumulativeUsage(currentChatResponseUsage,
						previousChatResponse);
				ChatResponse chatResponse = new ChatResponse(generations,
						from(chatCompletion, accumulatedUsage, rateLimit));

				observationContext.setResponse(chatResponse);

				return chatResponse;

			});

		return response;
	}

	@Override
	public Flux<ChatResponse> stream(Prompt prompt) {
		// 这里只准备 Prompt；真正的请求创建、观测和网络订阅在 internalStream 的 defer 中发生。
		Prompt requestPrompt = buildRequestPrompt(prompt);
		verifyPromptChatOptions(requestPrompt);
		return internalStream(requestPrompt);
	}

	/**
	 * Internal method to handle streaming chat completion calls with tool execution
	 * support.
	 * <p>
	 * 每次订阅独立建立 SDK 流、工具片段聚合状态和 Observation。
	 * 输出仍是增量 ChatResponse；聚合出的完整回答仅通过回调补充观测上下文。
	 * @param prompt the prompt for the chat completion
	 * @return a Flux of chat responses
	 */
	private Flux<ChatResponse> internalStream(Prompt prompt) {
		// 1. 订阅时读取 Reactor Context，避免提前捕获调用 stream() 时的线程上下文。
		return Flux.deferContextual(contextView -> {
			ChatCompletionCreateParams request = this.createRequest(prompt, true);
			RequestOptions requestOptions = this.buildRequestOptions(prompt);
			ConcurrentHashMap<String, String> roleMap = new ConcurrentHashMap<>();
			ConcurrentHashMap<String, String> reasoningMap = new ConcurrentHashMap<>();
			final ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
				.prompt(prompt)
				.provider(AiProvider.OPENAI.value())
				.streaming(true)
				.build();
			Observation observation = ChatModelObservationDocumentation.CHAT_MODEL_OPERATION.observation(
					this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry);
			Observation parentObservation = contextView.getOrDefault(ObservationThreadLocalAccessor.KEY, null);
			// 流可能在不同线程订阅，因此显式使用 Context 中的父观测，并仅在 start 时短暂打开父 Scope。
			observation.parentObservation(parentObservation);
			// Briefly make the parent observation current while starting this one, so
			// Micrometer tracing derives the span's parent from the parent observation
			// rather
			// than from whatever scope happens to be open on the current thread (e.g. the
			// servlet HTTP span). This keeps span parenting correct without relying on
			// automatic context propagation.
			try (Observation.Scope ignored = parentObservation != null ? parentObservation.openScope()
					: Observation.Scope.NOOP) {
				observation.start();
			}

			// 2. 把 SDK 回调桥接成 Flux。此处没有指定 Scheduler，回调线程由 SDK/HTTP 客户端决定。
			// Convert from AsyncStreamResponse<ChatCompletionChunk> to Flux<CCC>
			Flux<ChatCompletionChunk> chunks = Flux.<ChatCompletionChunk>create(sink -> {
				AsyncStreamResponse<ChatCompletionChunk> response = this.openAiClientAsync.chat()
					.completions()
					.createStreaming(request, requestOptions);
				// 正常终止、异常终止或下游取消都会释放这次流响应；不会关闭可复用的 SDK 客户端。
				sink.onDispose(response::close);
				response.subscribe(sink::next).onCompleteFuture().whenComplete((unused, throwable) -> {
					if (throwable != null) {
						sink.error(throwable);
					}
					else {
						sink.complete();
					}
				});
			});

			// 3. 普通文本片段立即下传；进入工具调用后缓冲到 TOOL_CALLS 结束标记，拼出完整参数 JSON。
			// 开始/结束判定查看第一个 choice，缓冲区内的合并则按各 choice.index 分组。
			// Next, aggregate CCCs that deal with tool calls together
			AtomicBoolean isInsideTool = new AtomicBoolean(false);
			Flux<ChatCompletion> aggregatedChatCompletions = chunks.doOnNext(chunk -> {
				if (ChunkMerger.hasToolCall(chunk)) {
					isInsideTool.set(true);
				}
			}).bufferUntil(chunk -> {
				if (isInsideTool.get() && ChunkMerger.toolCallsDone(chunk)) {
					isInsideTool.set(false);
					return true;
				}
				return !isInsideTool.get();
			}).map(ChunkMerger::mergeChunks).map(ChunkMerger::chunkToChatCompletion);

			// 4. 统一复用同步响应映射。role 可能只在前面的片段出现，reasoning 则跨片段累加。
			Flux<ChatResponse> chatResponses = aggregatedChatCompletions.map(chatCompletion -> {
				String id = chatCompletion.id();
				List<Generation> generations = chatCompletion.choices().stream().map(choice -> {
					roleMap.putIfAbsent(id, choice.message()._role().asString().isPresent()
							? choice.message()._role().asStringOrThrow() : "");

					// Accumulate reasoning fragments across the streamed chunks so the
					// final response of this stream carries the full reasoning content,
					// surviving last-wins metadata aggregation (e.g. MessageAggregator).
					String accumulatedReasoning = reasoningMap.merge(id + ":" + choice.index(),
							getReasoningContent(choice), String::concat);

					Map<String, Object> metadata = Map.of("id", id, //
							"role", roleMap.getOrDefault(id, ""), //
							"index", choice.index(), //
							"finishReason", choice.finishReason().value(), //
							"refusal", choice.message().refusal().orElse(""), //
							"annotations", choice.message().annotations().orElseGet(List::of), //
							REASONING_CONTENT, accumulatedReasoning //
					);

					return buildGeneration(choice, metadata, request);
				}).toList();
				Optional<CompletionUsage> usage = chatCompletion.usage();
				CompletionUsage usageVal = usage.orElse(null);
				Usage currentUsage = usageVal != null ? getDefaultUsage(usageVal) : new EmptyUsage();
				Usage accumulated = UsageCalculator.getCumulativeUsage(currentUsage, null);
				// Reading the rate limit here would mean the raw streaming call, whose
				// blocking StreamResponse would hold a worker for the whole stream.
				return new ChatResponse(generations, from(chatCompletion, accumulated, new EmptyRateLimit()));

			});

			// 5. 错误继续向订阅者传播；doFinally 覆盖完成、异常和取消，保证已启动的观测结束。
			// contextWrite 供上游订阅链读取当前模型 Observation，不表示所有回调自动处在同一线程 Scope。
			Flux<ChatResponse> observedResponses = chatResponses.doOnError(observation::error)
				.doFinally(s -> observation.stop())
				.contextWrite(ctx -> ctx.put(ObservationThreadLocalAccessor.KEY, observation));

			// MessageAggregator 旁路累积消息，正常完成时回填完整响应；不会把增量 Flux 改成单条最终结果。
			return new MessageAggregator().aggregate(observedResponses, observationContext::setResponse);

		});
	}

	/**
	 * 将 SDK 的一个候选回答还原为框架消息。工具调用保留 id、名称和原始参数 JSON，
	 * 供上层选择是否执行；这里只做数据映射，工具回调不会在此触发。
	 */
	private Generation buildGeneration(ChatCompletion.Choice choice, Map<String, Object> metadata,
			ChatCompletionCreateParams request) {
		ChatCompletionMessage message = choice.message();
		Map<String, Object> assistantMessageMetadata = new LinkedHashMap<>(metadata);
		// 兼容服务的工具调用扩展字段按调用 id 保存；历史消息再次发送时从 metadata 恢复到协议字段。
		Map<String, String> toolCallAdditionalProperties = extractToolCallAdditionalProperties(message);
		if (!toolCallAdditionalProperties.isEmpty()) {
			assistantMessageMetadata.put(TOOL_CALL_ADDITIONAL_PROPERTIES_METADATA_KEY, toolCallAdditionalProperties);
		}
		List<AssistantMessage.ToolCall> toolCalls = message.toolCalls()
			.map(list -> list.stream().filter(tc -> tc.function().isPresent()).map(tc -> {
				var opt = tc.function();
				if (opt.isEmpty()) {
					return null;
				}
				var funcCall = opt.get();
				var functionDef = funcCall.function();
				String id = funcCall.id();
				String name = functionDef.name();
				String arguments = functionDef.arguments();
				return new AssistantMessage.ToolCall(id, "function", name, arguments);
			}).filter(Objects::nonNull).toList())
			.orElse(List.of());

		ChatCompletion.Choice.FinishReason.Value finishReasonValue = choice.finishReason().value();
		var generationMetadataBuilder = ChatGenerationMetadata.builder()
			.finishReason(finishReasonValue != ChatCompletion.Choice.FinishReason.Value._UNKNOWN
					? finishReasonValue.name() : null);

		String textContent = message.content().orElse("");

		List<Media> media = new ArrayList<>();

		// 音频输出解码成 Media；正文缺失时使用 transcript，音频 id/过期时间归入生成级元数据。
		if (message.audio().isPresent() && StringUtils.hasText(message.audio().get().data())
				&& request.audio().isPresent()) {
			var audioOutput = message.audio().get();
			String mimeType = String.format("audio/%s",
					request.audio().get().format().value().name().toLowerCase(Locale.ROOT));
			byte[] audioData = Base64.getDecoder().decode(audioOutput.data());
			Resource resource = new ByteArrayResource(audioData);
			Media.builder().mimeType(MimeTypeUtils.parseMimeType(mimeType)).data(resource).id(audioOutput.id()).build();
			media.add(Media.builder()
				.mimeType(MimeTypeUtils.parseMimeType(mimeType))
				.data(resource)
				.id(audioOutput.id())
				.build());
			if (!StringUtils.hasText(textContent)) {
				textContent = audioOutput.transcript();
			}
			generationMetadataBuilder.metadata("audioId", audioOutput.id());
			generationMetadataBuilder.metadata("audioExpiresAt", audioOutput.expiresAt());
		}

		// Unwrap Optional values so downstream repositories (Neo4j, MongoDB, JDBC,
		// etc.) can serialize the metadata map without failing on java.util.Optional.
		assistantMessageMetadata
			.replaceAll((key, value) -> value instanceof Optional<?> optional ? optional.orElse(null) : value);

		var assistantMessage = AssistantMessage.builder()
			.content(textContent)
			.properties(assistantMessageMetadata)
			.toolCalls(toolCalls)
			.media(media)
			.build();
		return new Generation(assistantMessage, generationMetadataBuilder.build());
	}

	private Map<String, String> extractToolCallAdditionalProperties(ChatCompletionMessage message) {
		Map<String, String> result = new LinkedHashMap<>();
		message.toolCalls()
			.ifPresent(toolCalls -> toolCalls.forEach(toolCall -> toolCall.function().ifPresent(functionToolCall -> {
				Map<String, JsonValue> props = functionToolCall._additionalProperties();
				if (!CollectionUtils.isEmpty(props)) {
					try {
						result.put(functionToolCall.id(), objectMapper.writeValueAsString(props));
					}
					catch (JsonProcessingException ex) {
						throw new RuntimeException(ex);
					}
				}
			})));
		return result;
	}

	private ChatResponseMetadata from(ChatCompletion result, Usage usage, RateLimit rateLimit) {
		Assert.notNull(result, "OpenAI ChatCompletion must not be null");
		result.model();
		result.id();
		ChatResponseMetadata.Builder metadataBuilder = ChatResponseMetadata.builder()
			.id(result.id())
			.usage(usage)
			.model(result.model())
			.rateLimit(rateLimit)
			.keyValue("created", getCreated(result));

		result._additionalProperties().forEach((key, jsonValue) -> {
			try {
				Object value = JacksonUtils.getDefaultJsonMapper().convertValue(jsonValue, Object.class);
				metadataBuilder.keyValue(key, value);
			}
			catch (Exception e) {
				if (logger.isErrorEnabled()) {
					logger.error("Error parsing JSON value for key '" + key + "': " + jsonValue, e);
				}
				metadataBuilder.keyValue(key, jsonValue);
			}
		});

		return metadataBuilder.build();
	}

	/**
	 * Extract the created timestamp from a ChatCompletion result, returning 0 if the
	 * field is absent. Some OpenAI-compatible providers (e.g. GitHub Copilot) do not
	 * include the created field in their response.
	 */
	private long getCreated(ChatCompletion result) {
		try {
			return result.created();
		}
		catch (OpenAIInvalidDataException ex) {
			return 0L;
		}
	}

	private DefaultUsage getDefaultUsage(CompletionUsage usage) {
		Long cacheRead = usage.promptTokensDetails().flatMap(details -> details.cachedTokens()).orElse(null);
		return new DefaultUsage(toIntTokenCount("promptTokens", usage.promptTokens()),
				toIntTokenCount("completionTokens", usage.completionTokens()),
				toIntTokenCount("totalTokens", usage.totalTokens()), usage, cacheRead, null);
	}

	/**
	 * Narrows a server-supplied token count to an {@code int}. Usage counts come from the
	 * deserialised upstream response, so an out-of-range value means the response is not
	 * trustworthy; fail clearly with the offending field and value rather than either
	 * silently truncating the count (which would corrupt downstream cost/usage tracking
	 * with a plausible-looking but wrong number) or letting a bare
	 * {@link ArithmeticException} propagate.
	 * @param fieldName the name of the usage field being converted, for diagnostics
	 * @param value the upstream token count
	 * @return the value narrowed to an {@code int}
	 * @throws IllegalStateException if {@code value} is outside the {@code int} range
	 */
	private static int toIntTokenCount(String fieldName, long value) {
		try {
			return Math.toIntExact(value);
		}
		catch (ArithmeticException ex) {
			throw new IllegalStateException(
					"OpenAI-compatible provider returned an out-of-range " + fieldName + " value: " + value, ex);
		}
	}

	private void verifyPromptChatOptions(Prompt prompt) {
		var chatOptions = prompt.getOptions();

		if (chatOptions != null && chatOptions.getTopK() != null) {
			// OpenAI 请求构建器不写 topK；这里只告警，不替用户改写 Prompt 中的选项。
			logger.warn("The topK option is not supported by OpenAI chat models. Ignoring.");
		}
	}

	/**
	 * Creates a chat completion request from the given prompt.
	 * <p>
	 * 协议映射分三部分：消息角色与多模态内容、模型生成参数、工具定义及兼容服务扩展字段。
	 * 本方法只构造 SDK 参数；连接地址和凭据属于已创建客户端，timeout 另外进入 RequestOptions。
	 * @param prompt the prompt containing messages and options
	 * @param stream whether this is a streaming request
	 * @return the chat completion create parameters
	 */
	ChatCompletionCreateParams createRequest(Prompt prompt, boolean stream) {

		// buildRequestPrompt 只补全 null；直接调用模型时，非 null 的 options 必须是本供应商类型。
		OpenAiChatOptions requestOptions = (OpenAiChatOptions) prompt.getOptions();
		Assert.state(requestOptions != null, "ChatOptions must not be null");

		// Replaying reasoning content is what OpenAI-compatible reasoning endpoints such
		// as DeepSeek's thinking mode require, and what Groq rejects outright, so it can
		// be turned off per request. Absent an explicit opt-out the content is replayed
		// whenever present, which is the behavior plain OpenAI is unaffected by.
		boolean replayReasoningContent = !(prompt.getOptions() instanceof OpenAiChatOptions chatOptions)
				|| !Boolean.FALSE.equals(chatOptions.getReplayReasoningContent());

		// 第一阶段：按原有顺序转换消息。一条 TOOL 消息可能含多个结果，因此先返回列表，最后展平。
		List<ChatCompletionMessageParam> chatCompletionMessageParams = prompt.getInstructions()
			.stream()
			.map(message -> {
				if (message.getMessageType() == MessageType.USER || message.getMessageType() == MessageType.SYSTEM) {
					// USER/SYSTEM 共用 SDK 的 user 参数构建器，下面显式设置 wire role 保留实际角色。
					// Handle simple text content for user and system messages
					ChatCompletionUserMessageParam.Builder builder = ChatCompletionUserMessageParam.builder();

					if (message instanceof UserMessage userMessage
							&& !CollectionUtils.isEmpty(userMessage.getMedia())) {
						// 多模态仅在 UserMessage 分支读取；文本和各媒体成为同一条消息中的 content parts。
						// Handle media content (images, audio, files)
						List<ChatCompletionContentPart> parts = new ArrayList<>();

						String messageText = message.getText();
						if (messageText != null && !messageText.isEmpty()) {
							parts.add(ChatCompletionContentPart
								.ofText(ChatCompletionContentPartText.builder().text(messageText).build()));
						}

						// Add media content parts
						userMessage.getMedia().forEach(media -> {
							String mimeType = media.getMimeType().toString();
							if (mimeType.startsWith("image/")) {
								// URL 直接传递，byte[] 编成 data URL；框架不会在此主动下载远程图片。
								if (media.getData() instanceof java.net.URI uri) {
									parts.add(ChatCompletionContentPart
										.ofImageUrl(ChatCompletionContentPartImage.builder()
											.imageUrl(ChatCompletionContentPartImage.ImageUrl.builder()
												.url(uri.toString())
												.build())
											.build()));
								}
								else if (media.getData() instanceof String text) {
									// The org.springframework.ai.content.Media object
									// should store the URL as a java.net.URI but it
									// transforms it to String somewhere along the way,
									// for example in its Builder class. So, we accept
									// String as well here for image URLs.
									parts.add(ChatCompletionContentPart
										.ofImageUrl(ChatCompletionContentPartImage.builder()
											.imageUrl(
													ChatCompletionContentPartImage.ImageUrl.builder().url(text).build())
											.build()));
								}
								else if (media.getData() instanceof byte[] bytes) {
									// Assume the bytes are an image. So, convert the
									// bytes to a base64 encoded
									ChatCompletionContentPartImage.ImageUrl.Builder imageUrlBuilder = ChatCompletionContentPartImage.ImageUrl
										.builder();

									imageUrlBuilder.url("data:" + mimeType + ";base64,"
											+ Base64.getEncoder().encodeToString(bytes));
									parts.add(ChatCompletionContentPart
										.ofImageUrl(ChatCompletionContentPartImage.builder()
											.imageUrl(imageUrlBuilder.build())
											.build()));
								}
								else {
									if (logger.isInfoEnabled()) {
										logger.info("Could not process image media with data of type: "
												+ media.getData().getClass().getSimpleName()
												+ ". Only java.net.URI is supported for image URLs.");
									}
								}
							}
							else if (mimeType.startsWith("audio/")) {
								// 当前转换要求音频数据是 byte[]；MIME 含 mp3 时标为 MP3，其余此分支标为 WAV。
								parts.add(ChatCompletionContentPart
									.ofInputAudio(ChatCompletionContentPartInputAudio.builder()
										.inputAudio(ChatCompletionContentPartInputAudio.builder()
											.inputAudio(ChatCompletionContentPartInputAudio.InputAudio.builder()
												.data(fromAudioData(media.getData()))
												.format(mimeType.contains("mp3")
														? ChatCompletionContentPartInputAudio.InputAudio.Format.MP3
														: ChatCompletionContentPartInputAudio.InputAudio.Format.WAV)
												.build())
											.build()
											.inputAudio())
										.build()));
							}
							else if ("application/pdf".equals(mimeType)) {
								// PDF 使用 file 内容块；其他未专门处理的 MIME 在后续分支转成文本形式的数据 URL。
								parts.add(ChatCompletionContentPart.ofFile(ChatCompletionContentPart.File.builder()
									.file(ChatCompletionContentPart.File.FileObject.builder()
										.filename(media.getName())
										.fileData(fromMediaData(media.getMimeType(), media.getData()))
										.build())
									.build()));
							}
							else {
								// Assume it's a file or other media type represented as a
								// data URL
								parts.add(ChatCompletionContentPart.ofText(ChatCompletionContentPartText.builder()
									.text(fromMediaData(media.getMimeType(), media.getData()))
									.build()));
							}
						});
						builder.contentOfArrayOfContentParts(parts);
					}
					else {
						// Simple text message
						String messageText = message.getText();
						if (messageText != null) {
							builder.content(ChatCompletionContentPartText.builder().text(messageText).build().text());
						}
					}

					if (message.getMessageType() == MessageType.USER) {
						builder.role(JsonValue.from(MessageType.USER.getValue()));
					}
					else {
						builder.role(JsonValue.from(MessageType.SYSTEM.getValue()));
					}

					return List.of(ChatCompletionMessageParam.ofUser(builder.build()));
				}
				else if (message.getMessageType() == MessageType.ASSISTANT) {
					// 回放历史 AssistantMessage：文本、工具调用和 reasoning 扩展字段都参与下一轮请求。
					var assistantMessage = (AssistantMessage) message;
					ChatCompletionAssistantMessageParam.Builder builder = ChatCompletionAssistantMessageParam.builder()
						.role(JsonValue.from(MessageType.ASSISTANT.getValue()));

					if (assistantMessage.getText() != null) {
						builder.content(ChatCompletionAssistantMessageParam.builder()
							.content(assistantMessage.getText())
							.build()
							.content());
					}

					if (!CollectionUtils.isEmpty(assistantMessage.getToolCalls())) {
						Map<String, String> toolCallAdditionalProperties = toolCallAdditionalPropertiesFromMetadata(
								assistantMessage);

						List<ChatCompletionMessageToolCall> toolCalls = assistantMessage.getToolCalls()
							.stream()
							.map(toolCall -> {
								ChatCompletionMessageFunctionToolCall.Builder toolCallBuilder = ChatCompletionMessageFunctionToolCall
									.builder()
									.id(toolCall.id())
									.function(ChatCompletionMessageFunctionToolCall.Function.builder()
										.name(toolCall.name())
										.arguments(toolCall.arguments())
										.build());

								String jsonProps = toolCallAdditionalProperties.get(toolCall.id());
								if (StringUtils.hasText(jsonProps)) {
									Map<String, JsonValue> additionalProperties = new LinkedHashMap<>();
									try {
										objectMapper.readValue(jsonProps, MAP_TYPE_REF)
											.forEach((k, v) -> additionalProperties.put(k, JsonValue.from(v)));
									}
									catch (JsonProcessingException ex) {
										throw new IllegalStateException("Conversion from JSON to %s failed"
											.formatted(MAP_TYPE_REF.getType().getTypeName()), ex);
									}
									toolCallBuilder.putAllAdditionalProperties(additionalProperties);
								}
								return ChatCompletionMessageToolCall.ofFunction(toolCallBuilder.build());
							})
							.toList();

						builder.toolCalls(toolCalls);
					}

					// Replay reasoning content only when present and not opted out -
					// plain OpenAI is unaffected
					Object reasoningContent = assistantMessage.getMetadata().get(REASONING_CONTENT);
					if (replayReasoningContent && reasoningContent instanceof String reasoning
							&& StringUtils.hasText(reasoning)) {
						// "reasoning_content" is the wire field; REASONING_CONTENT is the
						// metadata key
						builder.putAdditionalProperty("reasoning_content", JsonValue.from(reasoning));
					}

					return List.of(ChatCompletionMessageParam.ofAssistant(builder.build()));
				}
				else if (message.getMessageType() == MessageType.TOOL) {
					// 每个执行结果以 tool_call_id 对应上一轮 assistant 的工具调用，不能只发送结果文本。
					ToolResponseMessage toolMessage = (ToolResponseMessage) message;

					ChatCompletionToolMessageParam.Builder builder = ChatCompletionToolMessageParam.builder();
					builder.content(toolMessage.getText() != null ? toolMessage.getText() : "");
					builder.role(JsonValue.from(MessageType.TOOL.getValue()));

					if (toolMessage.getResponses().isEmpty()) {
						return List.of(ChatCompletionMessageParam.ofTool(builder.build()));
					}
					return toolMessage.getResponses().stream().map(response -> {
						String callId = response.id();
						String callResponse = response.responseData();

						return ChatCompletionMessageParam
							.ofTool(builder.toolCallId(callId).content(callResponse).build());
					}).toList();
				}
				else {
					throw new IllegalArgumentException("Unsupported message type: " + message.getMessageType());
				}
			})
			.flatMap(List::stream)
			.toList();

		// 第二阶段：只写已设置的生成参数；deploymentName 优先于 model，适配部署名寻址的服务。
		ChatCompletionCreateParams.Builder builder = ChatCompletionCreateParams.builder();

		chatCompletionMessageParams.forEach(builder::addMessage);

		// Use deployment name if available (for Microsoft Foundry), otherwise use model
		// name
		if (requestOptions.getDeploymentName() != null) {
			builder.model(requestOptions.getDeploymentName());
		}
		else if (requestOptions.getModel() != null) {
			builder.model(requestOptions.getModel());
		}

		if (requestOptions.getFrequencyPenalty() != null) {
			builder.frequencyPenalty(requestOptions.getFrequencyPenalty());
		}
		if (requestOptions.getLogitBias() != null) {
			builder.logitBias(ChatCompletionCreateParams.LogitBias.builder()
				.putAllAdditionalProperties(requestOptions.getLogitBias()
					.entrySet()
					.stream()
					.collect(Collectors.toMap(Map.Entry::getKey, entry -> JsonValue.from(entry.getValue()))))
				.build());
		}
		if (requestOptions.getLogprobs() != null) {
			builder.logprobs(requestOptions.getLogprobs());
		}
		if (requestOptions.getTopLogprobs() != null) {
			builder.topLogprobs(requestOptions.getTopLogprobs());
		}
		if (requestOptions.getMaxTokens() != null) {
			builder.maxTokens(requestOptions.getMaxTokens());
		}
		if (requestOptions.getMaxCompletionTokens() != null) {
			builder.maxCompletionTokens(requestOptions.getMaxCompletionTokens());
		}
		if (requestOptions.getN() != null) {
			builder.n(requestOptions.getN());
		}
		if (requestOptions.getOutputModalities() != null) {
			builder.modalities(requestOptions.getOutputModalities()
				.stream()
				.map(modality -> ChatCompletionCreateParams.Modality.of(modality.toLowerCase(Locale.ROOT)))
				.toList());
		}
		if (requestOptions.getOutputAudio() != null) {
			builder.audio(requestOptions.getOutputAudio().toChatCompletionAudioParam());
		}
		if (requestOptions.getPresencePenalty() != null) {
			builder.presencePenalty(requestOptions.getPresencePenalty());
		}
		if (requestOptions.getResponseFormat() != null) {
			// 响应 JSON Schema 控制模型输出结构；它的 strict 与下面工具定义的 strict 是两个开关。
			ResponseFormat responseFormat = requestOptions.getResponseFormat();
			if (responseFormat.getType().equals(ResponseFormat.Type.TEXT)) {
				builder.responseFormat(ResponseFormatText.builder().build());
			}
			else if (responseFormat.getType().equals(ResponseFormat.Type.JSON_OBJECT)) {
				builder.responseFormat(ResponseFormatJsonObject.builder().build());
			}
			else if (responseFormat.getType().equals(ResponseFormat.Type.JSON_SCHEMA)) {
				String jsonSchemaString = responseFormat.getJsonSchema() != null ? responseFormat.getJsonSchema() : "";
				try {

					ResponseFormatJsonSchema.JsonSchema.Builder jsonSchemaBuilder = ResponseFormatJsonSchema.JsonSchema
						.builder();
					jsonSchemaBuilder.name("json_schema");
					Boolean strict = responseFormat.getStrict();
					jsonSchemaBuilder.strict(strict != null ? strict : true);

					ResponseFormatJsonSchema.JsonSchema.Schema schema = objectMapper.readValue(jsonSchemaString,
							ResponseFormatJsonSchema.JsonSchema.Schema.class);

					jsonSchemaBuilder.schema(schema);

					builder.responseFormat(
							ResponseFormatJsonSchema.builder().jsonSchema(jsonSchemaBuilder.build()).build());
				}
				catch (Exception e) {
					throw new IllegalArgumentException("Failed to parse JSON schema: " + jsonSchemaString, e);
				}
			}
			else {
				throw new IllegalArgumentException("Unsupported response format type: " + responseFormat.getType());
			}
		}
		if (requestOptions.getSeed() != null) {
			builder.seed(requestOptions.getSeed());
		}
		if (requestOptions.getStop() != null && !requestOptions.getStop().isEmpty()) {
			if (requestOptions.getStop().size() == 1) {
				builder.stop(ChatCompletionCreateParams.Stop.ofString(requestOptions.getStop().get(0)));
			}
			else {
				builder.stop(ChatCompletionCreateParams.Stop.ofStrings(requestOptions.getStop()));
			}
		}
		if (requestOptions.getTemperature() != null) {
			builder.temperature(requestOptions.getTemperature());
		}
		if (requestOptions.getTopP() != null) {
			builder.topP(requestOptions.getTopP());
		}
		if (requestOptions.getUser() != null) {
			builder.user(requestOptions.getUser());
		}
		if (requestOptions.getParallelToolCalls() != null) {
			builder.parallelToolCalls(requestOptions.getParallelToolCalls());
		}
		if (requestOptions.getReasoningEffort() != null) {
			builder.reasoningEffort(ReasoningEffort.of(requestOptions.getReasoningEffort().toLowerCase(Locale.ROOT)));
		}
		if (requestOptions.getVerbosity() != null) {
			builder.verbosity(ChatCompletionCreateParams.Verbosity.of(requestOptions.getVerbosity()));
		}

		if (requestOptions.getStore() != null) {
			builder.store(requestOptions.getStore());
		}
		if (requestOptions.getMetadata() != null && !requestOptions.getMetadata().isEmpty()) {
			builder.metadata(ChatCompletionCreateParams.Metadata.builder()
				.putAllAdditionalProperties(requestOptions.getMetadata()
					.entrySet()
					.stream()
					.collect(Collectors.toMap(Map.Entry::getKey, entry -> JsonValue.from(entry.getValue()))))
				.build());
		}
		if (requestOptions.getServiceTier() != null) {
			builder.serviceTier(ChatCompletionCreateParams.ServiceTier.of(requestOptions.getServiceTier()));
		}

		if (requestOptions.getPromptCacheKey() != null) {
			builder.promptCacheKey(requestOptions.getPromptCacheKey());
		}

		if (requestOptions.getCustomHeaders() != null && !requestOptions.getCustomHeaders().isEmpty()) {
			requestOptions.getCustomHeaders().forEach(builder::putAdditionalHeader);
		}

		if (stream) {
			// 默认请求流末尾的 usage；显式提供 streamOptions 时以该对象的字段值为准。
			if (requestOptions.getStreamOptions() != null) {
				ChatCompletionStreamOptions.Builder streamOptionsBuilder = ChatCompletionStreamOptions.builder();

				var ops = requestOptions.getStreamOptions();

				streamOptionsBuilder.includeObfuscation(ops.includeObfuscation() != null && ops.includeObfuscation());
				streamOptionsBuilder.includeUsage(ops.includeUsage() != null && ops.includeUsage());

				if (!CollectionUtils.isEmpty(ops.additionalProperties())) {
					Map<String, com.openai.core.JsonValue> nativeParams = ops.additionalProperties()
						.entrySet()
						.stream()
						.map(e -> Map.entry(e.getKey(), com.openai.core.JsonValue.from(e.getValue())))
						.collect(HashMap::new, (m, e) -> m.put(e.getKey(), e.getValue()), HashMap::putAll);

					streamOptionsBuilder.putAllAdditionalProperties(nativeParams);
				}
				builder.streamOptions(streamOptionsBuilder.build());
			}
			else {
				builder.streamOptions(ChatCompletionStreamOptions.builder()
					.includeUsage(true) // Include usage by default for streaming
					.build());
			}
		}

		// 第三阶段：解析 ToolCallback/工具名对应的定义，只发送名称、描述和输入 Schema。
		// 此处调用 resolveToolDefinitions，不调用 executeToolCalls；工具执行属于上层 Advisor。
		// Add the tool definitions to the request's tools parameter.
		List<ToolDefinition> toolDefinitions = this.toolCallingManager.resolveToolDefinitions(requestOptions);
		if (!CollectionUtils.isEmpty(toolDefinitions)) {
			builder.tools(getChatCompletionTools(toolDefinitions, requestOptions));
		}

		if (requestOptions.getToolChoice() != null) {
			if (requestOptions.getToolChoice() instanceof ChatCompletionToolChoiceOption toolChoiceOption) {
				builder.toolChoice(toolChoiceOption);
			}
			else if (requestOptions.getToolChoice() instanceof String json) {
				if (json.equals("auto")) {
					builder.toolChoice(ChatCompletionToolChoiceOption.ofAuto(ChatCompletionToolChoiceOption.Auto.AUTO));
				}
				else if (json.equals("none")) {
					builder.toolChoice(ChatCompletionToolChoiceOption.ofAuto(ChatCompletionToolChoiceOption.Auto.NONE));
				}
				else if (json.equals("required")) {
					builder.toolChoice(
							ChatCompletionToolChoiceOption.ofAuto(ChatCompletionToolChoiceOption.Auto.REQUIRED));
				}
				else {
					try {
						var node = JacksonUtils.getDefaultJsonMapper().readTree(json);
						builder.toolChoice(parseToolChoice(node));
					}
					catch (Exception e) {
						throw new IllegalArgumentException("Failed to parse toolChoice JSON: " + json, e);
					}
				}
			}
		}

		// Add extraBody parameters as additional body properties for OpenAI-compatible
		// providers
		if (requestOptions.getExtraBody() != null && !requestOptions.getExtraBody().isEmpty()) {
			Map<String, JsonValue> extraParams = requestOptions.getExtraBody()
				.entrySet()
				.stream()
				.collect(Collectors.toMap(Map.Entry::getKey, entry -> JsonValue.from(entry.getValue())));
			builder.additionalBodyProperties(extraParams);
		}

		return builder.build();
	}

	/**
	 * Creates a RequestOptions instance from the given prompt.
	 * <p>
	 * 请求级 timeout 进入 SDK RequestOptions；未设置时让 SDK 使用客户端默认值。
	 * 其他连接参数没有在此重新建客户端，运行时 options 不会重配连接地址、凭据或重试次数。
	 * @param prompt the prompt containing messages and options
	 * @return a RequestOptions instance
	 */
	private RequestOptions buildRequestOptions(Prompt prompt) {
		Assert.notNull(prompt, "Prompt cannot be null");
		Assert.isInstanceOf(OpenAiChatOptions.class, prompt.getOptions(),
				"Prompt options must be OpenAiChatOptions type");
		OpenAiChatOptions chatOptions = (OpenAiChatOptions) prompt.getOptions();
		RequestOptions.Builder requestOptionsBuilder = RequestOptions.builder();
		if (chatOptions.getTimeout() != null) {
			requestOptionsBuilder.timeout(chatOptions.getTimeout());
		}
		return requestOptionsBuilder.build();
	}

	private Map<String, String> toolCallAdditionalPropertiesFromMetadata(AssistantMessage assistantMessage) {
		Object value = assistantMessage.getMetadata().get(TOOL_CALL_ADDITIONAL_PROPERTIES_METADATA_KEY);
		if (!(value instanceof Map<?, ?> rawMap)) {
			return Map.of();
		}
		Map<String, String> result = new LinkedHashMap<>();
		rawMap.forEach((k, v) -> {
			if (k instanceof String id && v instanceof String json) {
				result.put(id, json);
			}
		});
		return result;
	}

	public static ChatCompletionToolChoiceOption parseToolChoice(JsonNode node) {
		String type = node.get("type").asString();
		switch (type) {
			case "function":
				String functionName = node.get("function").get("name").asString();
				ChatCompletionNamedToolChoice.Function func = ChatCompletionNamedToolChoice.Function.builder()
					.name(functionName)
					.build();
				ChatCompletionNamedToolChoice named = ChatCompletionNamedToolChoice.builder().function(func).build();
				return ChatCompletionToolChoiceOption.ofNamedToolChoice(named);
			case "auto":
				// There is a built-in “auto” option — but how to get it depends on SDK
				// version
				return ChatCompletionToolChoiceOption.ofAuto(ChatCompletionToolChoiceOption.Auto.AUTO);
			case "required":
				return ChatCompletionToolChoiceOption.ofAuto(ChatCompletionToolChoiceOption.Auto.REQUIRED);
			case "none":
				return ChatCompletionToolChoiceOption.ofAuto(ChatCompletionToolChoiceOption.Auto.NONE);
			default:
				throw new IllegalArgumentException("Unknown tool_choice type: " + type);
		}
	}

	private String fromAudioData(Object audioData) {
		if (audioData instanceof byte[] bytes) {
			return Base64.getEncoder().encodeToString(bytes);
		}
		throw new IllegalArgumentException("Unsupported audio data type: " + audioData.getClass().getSimpleName());
	}

	private String fromMediaData(org.springframework.util.MimeType mimeType, Object mediaContentData) {
		if (mediaContentData instanceof byte[] bytes) {
			// Assume the bytes are an image. So, convert the bytes to a base64 encoded
			// following the prefix pattern.
			return String.format("data:%s;base64,%s", mimeType.toString(), Base64.getEncoder().encodeToString(bytes));
		}
		else if (mediaContentData instanceof String text) {
			// Assume the text is a URLs or a base64 encoded image prefixed by the user.
			return text;
		}
		else {
			throw new IllegalArgumentException(
					"Unsupported media data type: " + mediaContentData.getClass().getSimpleName());
		}
	}

	private List<ChatCompletionTool> getChatCompletionTools(List<ToolDefinition> toolDefinitions,
			@Nullable OpenAiChatOptions requestOptions) {
		// strict 默认关闭；显式启用后将常规 Schema 的可选字段改为 required + 可空类型。
		// 这里调整的是发送给模型的协议 Schema，不改变 Java 方法参数或本地工具执行方式。
		return toolDefinitions.stream().map(toolDefinition -> {
			FunctionParameters.Builder parametersBuilder = FunctionParameters.builder();
			// Defaults to false: OpenAI's strict mode requires every schema property to
			// appear in "required" (optionality is expressed via nullable types, not
			// omission), which JsonSchemaGenerator does not produce by default. When a
			// caller opts in via OpenAiChatOptions#strict, applyStrictModeRequirements
			// below rewrites the schema to satisfy that contract.
			Boolean strictMode = false;
			if (requestOptions != null && requestOptions.getStrict() != null) {
				strictMode = requestOptions.getStrict();
			}
			else if (this.options != null && this.options.getStrict() != null) {
				strictMode = this.options.getStrict();
			}
			if (!toolDefinition.inputSchema().isEmpty()) {
				// Parse the schema and add its properties directly
				try {
					@SuppressWarnings("unchecked")
					Map<String, Object> schemaMap = objectMapper.readValue(toolDefinition.inputSchema(), Map.class);

					if (Boolean.TRUE.equals(strictMode)) {
						applyStrictModeRequirements(schemaMap);
					}

					// Add each property from the schema to the parameters
					schemaMap
						.forEach((key, value) -> parametersBuilder.putAdditionalProperty(key, JsonValue.from(value)));
				}
				catch (Exception e) {
					// 当前契约是记录 Schema 转换错误后继续构建定义，异常不会在这里直接向外抛出。
					logger.error("Failed to parse tool schema", e);
				}
			}
			FunctionDefinition functionDefinition = FunctionDefinition.builder()
				.name(toolDefinition.name())
				.description(toolDefinition.description())
				.parameters(parametersBuilder.build())
				.strict(strictMode)
				.build();

			return ChatCompletionTool
				.ofFunction(ChatCompletionFunctionTool.builder().function(functionDefinition).build());
		}).toList();
	}

	/**
	 * A JSON Schema fragment representing exactly the {@code null} type, used as the
	 * extra {@code anyOf} branch when widening a {@code $ref}/{@code anyOf}-based
	 * property to accept {@code null}.
	 */
	private static final Map<String, Object> NULL_TYPE_SCHEMA = Map.of("type", "null");

	/**
	 * Rewrites an object schema in place to satisfy OpenAI's strict function-calling
	 * mode: every property must be listed in "required", with optionality expressed via a
	 * nullable type rather than omission from "required", and every object level must pin
	 * {@code "additionalProperties"} to {@code false}. {@link JsonSchemaGenerator}
	 * produces conventional JSON Schema instead - it omits
	 * {@code @ToolParam(required = false)} properties from "required" - so this widens
	 * the type of each such property to also accept {@code null} and backfills "required"
	 * with every property key. Schemas from other sources (MCP servers, hand-written
	 * inputSchema JSON) may also lack "additionalProperties", so it is backfilled with
	 * {@code false} wherever absent; an explicit value is preserved. Recurses into nested
	 * object/array-item schemas and {@code $defs} definitions so their own optional
	 * properties are fixed up too.
	 */
	@SuppressWarnings("unchecked")
	private static void applyStrictModeRequirements(Map<String, Object> schema) {
		if (schema.get("$defs") instanceof Map<?, ?> defs) {
			for (Object definition : defs.values()) {
				if (definition instanceof Map<?, ?> definitionSchema) {
					applyStrictModeRequirements((Map<String, Object>) definitionSchema);
				}
			}
		}

		if (!(schema.get("properties") instanceof Map<?, ?> properties)) {
			return;
		}

		schema.putIfAbsent("additionalProperties", false);

		if (properties.isEmpty()) {
			return;
		}

		List<?> alreadyRequired = schema.get("required") instanceof List<?> required ? required : List.of();

		for (Object propertySchemaValue : properties.values()) {
			if (!(propertySchemaValue instanceof Map<?, ?> propertySchema)) {
				continue;
			}
			applyStrictModeRequirements((Map<String, Object>) propertySchema);
			if (propertySchema.get("items") instanceof Map<?, ?> itemsSchema) {
				applyStrictModeRequirements((Map<String, Object>) itemsSchema);
			}
		}

		for (Map.Entry<?, ?> property : properties.entrySet()) {
			if (alreadyRequired.contains(property.getKey())
					|| !(property.getValue() instanceof Map<?, ?> propertySchema)) {
				continue;
			}
			widenToNullable((Map<String, Object>) propertySchema);
		}

		schema.put("required", new ArrayList<>(properties.keySet()));
	}

	/**
	 * Widens a property schema's "type" to also accept {@code null}. Properties defined
	 * purely via {@code $ref} or {@code anyOf} (no "type" key of their own - typically a
	 * complex object parameter) are instead wrapped in an {@code anyOf} alongside a null
	 * branch, since there is no "type" value to widen directly.
	 */
	private static void widenToNullable(Map<String, Object> propertySchema) {
		Object type = propertySchema.get("type");
		if (type instanceof String typeName) {
			if (!"null".equals(typeName)) {
				propertySchema.put("type", new ArrayList<>(List.of(typeName, "null")));
			}
			return;
		}
		if (type instanceof List<?> typeList) {
			if (!typeList.contains("null")) {
				List<Object> widened = new ArrayList<>(typeList);
				widened.add("null");
				propertySchema.put("type", widened);
			}
			return;
		}
		if (propertySchema.get("anyOf") instanceof List<?> anyOf) {
			if (!anyOf.contains(NULL_TYPE_SCHEMA)) {
				List<Object> widened = new ArrayList<>(anyOf);
				widened.add(NULL_TYPE_SCHEMA);
				propertySchema.put("anyOf", widened);
			}
			return;
		}
		if (propertySchema.containsKey("$ref")) {
			Map<String, Object> refBranch = new LinkedHashMap<>(propertySchema);
			Object description = refBranch.remove("description");
			propertySchema.clear();
			if (description != null) {
				propertySchema.put("description", description);
			}
			propertySchema.put("anyOf", new ArrayList<>(List.of(refBranch, NULL_TYPE_SCHEMA)));
		}
	}

	private String getReasoningContent(ChatCompletion.Choice choice) {
		String reasoningContent = "";
		Map<String, JsonValue> additionalProperties = choice.message()._additionalProperties();
		if (additionalProperties.get("reasoning_content") != null) {
			reasoningContent = (String) additionalProperties.get("reasoning_content").asString().orElse("");
		}
		else {
			if (additionalProperties.get("reasoning") != null) {
				reasoningContent = (String) additionalProperties.get("reasoning").asString().orElse("");
			}
		}
		return reasoningContent;
	}

	/**
	 * Use the provided convention for reporting observation data
	 * @param observationConvention The provided convention
	 */
	public void setObservationConvention(ChatModelObservationConvention observationConvention) {
		Assert.notNull(observationConvention, "observationConvention cannot be null");
		this.observationConvention = observationConvention;
	}

	/**
	 * Look at the options of the provided prompt. If none are provided, return a new
	 * prompt using this model {@link ChatModel#getOptions() options}. Otherwise, use the
	 * prompt as is.
	 * <p>
	 * 这里的优先级以整份对象为单位：Prompt 有 options 就全部使用它，不能理解为
	 * “运行时非空字段覆盖默认字段”。只调整部分参数时，应从模型 options 的 mutate() 开始。
	 */
	private Prompt buildRequestPrompt(Prompt prompt) {
		if (prompt.getOptions() == null) {
			return prompt.mutate().chatOptions(this.getOptions()).build();
		}
		else {
			return prompt;
		}
	}

	/**
	 * 工具调用的参数可能被拆在多个 SSE 增量中；此辅助类按 choice.index、toolCall.index
	 * 两级索引合并，保留首个非空 id/名称并串接 arguments，再转成普通响应供公共映射逻辑处理。
	 */
	static final class ChunkMerger {

		static boolean hasToolCall(ChatCompletionChunk chunk) {
			return !chunk.choices().isEmpty()
					&& chunk.choices().get(0).delta().toolCalls().filter(toolCalls -> !toolCalls.isEmpty()).isPresent();
		}

		static boolean toolCallsDone(ChatCompletionChunk chunk) {
			return !chunk.choices().isEmpty()
					&& FinishReason.TOOL_CALLS == chunk.choices().get(0).finishReason().orElse(null);
		}

		static ChatCompletionChunk mergeChunks(List<ChatCompletionChunk> chunks) {
			// 每个缓冲区以首片段为基底，后续片段更新 usage/serviceTier 并合并相同候选索引。
			ChatCompletionChunk.Builder builder = chunks.get(0).toBuilder();
			Map<Long, Choice> choices = new LinkedHashMap<>();
			chunks.get(0).choices().forEach(choice -> choices.put(choice.index(), choice));

			for (int i = 1; i < chunks.size(); i++) {
				ChatCompletionChunk chunk = chunks.get(i);
				chunk.usage().ifPresent(builder::usage);
				chunk.serviceTier().ifPresent(builder::serviceTier);
				chunk.choices()
					.forEach(choice -> choices.compute(choice.index(),
							(ix, c) -> c == null ? choice : mergeChoices(c, choice)));
			}
			return builder.choices(new ArrayList<>(choices.values())).build();
		}

		private static Choice mergeChoices(Choice c1, Choice c2) {
			return Choice.builder()
				.index(c1.index())
				.finishReason(c1.finishReason().or(c2::finishReason))
				.logprobs(c1.logprobs().or(c2::logprobs))
				.delta(mergeDeltas(c1.delta(), c2.delta()))
				.build();
		}

		private static Delta mergeDeltas(Delta left, Delta right) {
			// 工具索引才是片段归属依据；后续片段的 id 可能缺失或为空，不能用它划分新调用。
			// Deltas of the same logical tool call share the required 'index' field.
			// Some OpenAI-compatible providers (e.g. DeepSeek) send an empty-string id
			// on continuation deltas instead of omitting it, so id presence cannot be
			// used to detect the start of a new tool call.
			var tcs = Stream.of(left.toolCalls(), right.toolCalls()).flatMap(Optional::stream).reduce((tcs1, tcs2) -> {
				if (tcs2.isEmpty()) {
					return tcs1;
				}
				Map<Long, ToolCall> mergedByIndex = new LinkedHashMap<>();
				tcs1.forEach(tc -> mergedByIndex.merge(tc.index(), tc, ChunkMerger::mergeToolCalls));
				tcs2.forEach(tc -> mergedByIndex.merge(tc.index(), tc, ChunkMerger::mergeToolCalls));
				return List.copyOf(mergedByIndex.values());
			}).orElse(List.of());

			Delta.Builder deltaBuilder = left.toBuilder().toolCalls(tcs);
			// Concatenate reasoning fragments (e.g. DeepSeek "reasoning_content") so
			// they survive the tool-call chunk merge instead of keeping only the first
			// chunk's value.
			for (String reasoningKey : List.of("reasoning_content", "reasoning")) {
				stringProperty(right, reasoningKey).filter(StringUtils::hasLength)
					.ifPresent(rightFragment -> deltaBuilder.putAdditionalProperty(reasoningKey,
							JsonValue.from(stringProperty(left, reasoningKey).orElse("") + rightFragment)));
			}
			return deltaBuilder.build();
		}

		private static Optional<String> stringProperty(Delta delta, String key) {
			return Optional.ofNullable(delta._additionalProperties().get(key)).flatMap(JsonValue::asString);
		}

		private static ToolCall mergeToolCalls(ToolCall previous, ToolCall current) {
			// arguments 仍是 JSON 字符串片段，此处只拼接，不在 JSON 尚未完整时反序列化。
			String arguments = Stream
				.of(previous.function().flatMap(Function::arguments), current.function().flatMap(Function::arguments))
				.flatMap(Optional::stream)
				.collect(Collectors.joining());
			return previous.toBuilder()
				.id(firstWithText(previous.id(), current.id()))
				.putAllAdditionalProperties(current._additionalProperties())
				.function(previous.function()
					.map(Function::toBuilder)
					.orElseGet(Function::builder)
					.name(firstWithText(previous.function().flatMap(Function::name),
							current.function().flatMap(Function::name)))
					.arguments(arguments)
					.build())
				.build();
		}

		private static String firstWithText(Optional<String> first, Optional<String> second) {
			return first.filter(StringUtils::hasText).or(() -> second.filter(StringUtils::hasText)).orElse("");
		}

		/**
		 * Convert a ChatCompletionChunk into a ChatCompletion.
		 * <p>
		 * 转换前要求工具的 function、id 和 name 已完整；缺失时明确失败，
		 * 避免上层 Advisor 拿不完整的工具调用去解析执行。
		 */
		static ChatCompletion chunkToChatCompletion(ChatCompletionChunk chunk) {
			List<ChatCompletion.Choice> choices = chunk.choices().stream().map(cccc -> {
				ChatCompletion.Choice.Builder choiceBuilder = ChatCompletion.Choice.builder();

				choiceBuilder.index(cccc.index());

				choiceBuilder.finishReason(ChatCompletion.Choice.FinishReason.of(""));
				cccc.finishReason()
					.ifPresent(finishReason -> choiceBuilder.finishReason(ChatCompletion.Choice.FinishReason
						.of(finishReason.value().name().toLowerCase(Locale.ROOT))));

				if (cccc.logprobs().isPresent()) {
					var logprobs = cccc.logprobs().get();
					choiceBuilder.logprobs(ChatCompletion.Choice.Logprobs.builder()
						.content(logprobs.content())
						.refusal(logprobs.refusal())
						.build());
				}
				else {
					choiceBuilder.logprobs(
							ChatCompletion.Choice.Logprobs.builder().content(List.of()).refusal(List.of()).build());
				}

				ChatCompletionMessage.Builder msgBuilder = ChatCompletionMessage.builder()
					.content(cccc.delta().content())
					.refusal(cccc.delta().refusal())
					// Carry over provider-specific delta fields (e.g. reasoning_content)
					// so they are readable on the message, as in the non-streaming path.
					.putAllAdditionalProperties(cccc.delta()._additionalProperties());
				cccc.delta().toolCalls().ifPresent(ccctcs -> {
					msgBuilder.toolCalls(ccctcs.stream().map(tc -> {
						ChatCompletionMessageFunctionToolCall.Builder toolCallBuilder = ChatCompletionMessageFunctionToolCall
							.builder();
						Function function = tc.function()
							.orElseThrow(() -> new IllegalStateException("Tool call function is missing"));
						String id = tc.id()
							.filter(StringUtils::hasText)
							.orElseThrow(() -> new IllegalStateException("Tool call id is missing"));
						String name = function.name()
							.filter(StringUtils::hasText)
							.orElseThrow(() -> new IllegalStateException("Tool call function name is missing"));
						toolCallBuilder.putAllAdditionalProperties(tc._additionalProperties());
						toolCallBuilder.id(id);
						toolCallBuilder.function(ChatCompletionMessageFunctionToolCall.Function.builder()
							.name(name)
							.arguments(function.arguments().orElse(""))
							.build());
						return ChatCompletionMessageToolCall.ofFunction(toolCallBuilder.build());
					}).toList());
				});
				choiceBuilder.message(msgBuilder.build());
				return choiceBuilder.build();
			}).toList();

			return ChatCompletion.builder()
				.id(chunk.id())
				.choices(choices)
				.created(getCreated(chunk))
				.model(chunk.model())
				.usage(chunk.usage()
					.orElse(CompletionUsage.builder().promptTokens(0).completionTokens(0).totalTokens(0).build()))
				.putAllAdditionalProperties(chunk._additionalProperties())
				.build();
		}

		/**
		 * Extract the created timestamp from a ChatCompletionChunk, returning 0 if
		 * absent.
		 */
		private static long getCreated(ChatCompletionChunk chunk) {
			try {
				return chunk.created();
			}
			catch (OpenAIInvalidDataException ex) {
				return 0L;
			}
		}

	}

	/**
	 * Response format (text, json_object, json_schema) for OpenAiChatModel responses.
	 *
	 * @author Julien Dubois
	 * @author Mariusz Bernacki
	 * @author Grogdunn
	 * @author Thomas Vitale
	 * @author John Blum
	 * @author Mark Pollack
	 * @author Josh Long
	 * @author Jemin Huh
	 * @author Ueibin Kim
	 * @author Alexandros Pappas
	 * @author luocongqiu
	 * @author Hyunjoon Choi
	 * @author Jonghoon Park
	 * @author Sebastien Deleuze
	 * @author Bishen Yu
	 */
	public static class ResponseFormat {

		private Type type = Type.TEXT;

		private @Nullable String jsonSchema;

		private @Nullable Boolean strict;

		public Type getType() {
			return this.type;
		}

		public void setType(Type type) {
			this.type = type;
		}

		public @Nullable String getJsonSchema() {
			return this.jsonSchema;
		}

		public void setJsonSchema(@Nullable String jsonSchema) {
			this.jsonSchema = jsonSchema;
		}

		/**
		 * Whether to enable strict schema adherence for JSON schema response format.
		 * Defaults to {@code true} when unset.
		 * <p>
		 * This applies only to the JSON schema response format and is distinct from
		 * {@link OpenAiChatOptions.Builder#strict(Boolean)}, which controls strict mode
		 * for tool/function calling.
		 * @return the strict flag, or {@code null} if not configured
		 */
		public @Nullable Boolean getStrict() {
			return this.strict;
		}

		public void setStrict(@Nullable Boolean strict) {
			this.strict = strict;
		}

		@Override
		public boolean equals(@Nullable Object o) {
			if (this == o) {
				return true;
			}
			if (o == null || getClass() != o.getClass()) {
				return false;
			}
			ResponseFormat that = (ResponseFormat) o;
			return this.type == that.type && Objects.equals(this.jsonSchema, that.jsonSchema)
					&& Objects.equals(this.strict, that.strict);
		}

		@Override
		public int hashCode() {
			return Objects.hash(this.type, this.jsonSchema, this.strict);
		}

		public static Builder builder() {
			return new Builder();
		}

		public static final class Builder {

			private final ResponseFormat responseFormat = new ResponseFormat();

			private Builder() {
			}

			public Builder type(Type type) {
				this.responseFormat.setType(type);
				return this;
			}

			public Builder jsonSchema(String jsonSchema) {
				this.responseFormat.setType(Type.JSON_SCHEMA);
				this.responseFormat.setJsonSchema(jsonSchema);
				return this;
			}

			/**
			 * Whether to enable strict schema adherence for JSON schema response format.
			 * <p>
			 * Not to be confused with {@link OpenAiChatOptions.Builder#strict(Boolean)},
			 * which applies to tool/function calling rather than response format.
			 * @param strict the strict flag
			 * @return this builder
			 */
			public Builder strict(@Nullable Boolean strict) {
				this.responseFormat.setStrict(strict);
				return this;
			}

			public ResponseFormat build() {
				return this.responseFormat;
			}

		}

		public enum Type {

			/**
			 * Generates a text response. (default)
			 */
			TEXT,

			/**
			 * Enables JSON mode, which guarantees the message the model generates is
			 * valid JSON.
			 */
			JSON_OBJECT,

			/**
			 * Enables Structured Outputs which guarantees the model will match your
			 * supplied JSON schema.
			 */
			JSON_SCHEMA

		}

	}

	/**
	 * Builder for creating {@link OpenAiChatModel} instances.
	 * <p>
	 * 模型持有可复用的同步与异步客户端。未显式传入的客户端才由 OpenAiSetup 创建，
	 * 此时连接、重试、HTTP 观测和 customizer 生效；传入客户端则沿用调用方配置。
	 */
	public static final class Builder {

		private @Nullable OpenAIClient openAiClient;

		private @Nullable OpenAIClientAsync openAiClientAsync;

		private @Nullable OpenAiChatOptions options;

		private @Nullable ToolCallingManager toolCallingManager;

		private @Nullable ObservationRegistry observationRegistry;

		private @Nullable MeterRegistry meterRegistry;

		private List<OpenAiHttpClientBuilderCustomizer> httpClientCustomizers = new ArrayList<>();

		private Builder() {
		}

		/**
		 * Sets the synchronous OpenAI client.
		 * @param openAiClient the synchronous client
		 * @return this builder
		 */
		public Builder openAiClient(OpenAIClient openAiClient) {
			this.openAiClient = openAiClient;
			return this;
		}

		/**
		 * Sets the asynchronous OpenAI client.
		 * @param openAiClientAsync the asynchronous client
		 * @return this builder
		 */
		public Builder openAiClientAsync(OpenAIClientAsync openAiClientAsync) {
			this.openAiClientAsync = openAiClientAsync;
			return this;
		}

		/**
		 * Sets the chat options.
		 * @param options the chat options
		 * @return this builder
		 */
		public Builder options(OpenAiChatOptions options) {
			this.options = options;
			return this;
		}

		/**
		 * Sets the tool calling manager used for internal tool execution.
		 * <p>
		 * 当前模型实际只使用该管理器解析工具定义；本地执行与多轮调用已由上层 Advisor 负责。
		 * @param toolCallingManager the tool calling manager
		 * @return this builder
		 * @deprecated since 2.0.0 for removal in 3.0.0 — internal tool execution in
		 * {@link OpenAiChatModel} is superseded by {@code ToolCallingAdvisor} used via
		 * {@code ChatClient}.
		 */
		@Deprecated(since = "2.0.0", forRemoval = true)
		public Builder toolCallingManager(ToolCallingManager toolCallingManager) {
			this.toolCallingManager = toolCallingManager;
			return this;
		}

		/**
		 * Sets the observation registry for metrics and tracing.
		 * @param observationRegistry the observation registry
		 * @return this builder
		 */
		public Builder observationRegistry(ObservationRegistry observationRegistry) {
			this.observationRegistry = observationRegistry;
			return this;
		}

		public Builder meterRegistry(@Nullable MeterRegistry meterRegistry) {
			this.meterRegistry = meterRegistry;
			return this;
		}

		/**
		 * Registers an {@link OpenAiHttpClientBuilderCustomizer} that mutates the
		 * underlying OkHttp client builder before the OpenAI clients are constructed. Use
		 * this to attach OkHttp interceptors (e.g. OAuth2 bearer-token injection), swap
		 * the dispatcher executor, or tweak any other OkHttp setting. Customizers are
		 * applied in the order they are registered, after Spring AI's own defaults, so
		 * user code wins.
		 */
		public Builder httpClientBuilderCustomizer(OpenAiHttpClientBuilderCustomizer customizer) {
			Assert.notNull(customizer, "customizer cannot be null");
			this.httpClientCustomizers.add(customizer);
			return this;
		}

		/**
		 * Sets the full list of {@link OpenAiHttpClientBuilderCustomizer customizers} to
		 * apply, replacing any customizers registered earlier on this builder. The order
		 * of the list is preserved when invoking the customizers.
		 */
		public Builder httpClientBuilderCustomizers(List<OpenAiHttpClientBuilderCustomizer> customizers) {
			Assert.notNull(customizers, "customizers cannot be null");
			this.httpClientCustomizers = new ArrayList<>(customizers);
			return this;
		}

		/**
		 * Builds a new {@link OpenAiChatModel} instance.
		 * @return the configured chat model
		 */
		public OpenAiChatModel build() {
			// 先确定模型默认选项与观测注册表，再按需构造两种客户端；构建模型不会发起推理请求。
			OpenAiChatOptions resolvedOptions = Objects.requireNonNullElseGet(this.options,
					() -> OpenAiChatOptions.builder().build());
			ObservationRegistry resolvedObservationRegistry = Objects.requireNonNullElse(this.observationRegistry,
					ObservationRegistry.NOOP);

			OpenAIClient resolvedClient = Objects.requireNonNullElseGet(this.openAiClient,
					() -> OpenAiSetup.setupSyncClient(resolvedOptions.getBaseUrl(), resolvedOptions.getApiKey(),
							resolvedOptions.getCredential(), resolvedOptions.getMicrosoftDeploymentName(),
							resolvedOptions.getMicrosoftFoundryServiceVersion(), resolvedOptions.getOrganizationId(),
							resolvedOptions.isMicrosoftFoundry(), resolvedOptions.isGitHubModels(),
							resolvedOptions.getModel(),
							Objects.requireNonNullElse(resolvedOptions.getTimeout(),
									AbstractOpenAiOptions.DEFAULT_TIMEOUT),
							resolvedOptions.getMaxRetries(), resolvedOptions.getProxy(),
							resolvedOptions.getCustomHeaders(), resolvedObservationRegistry, this.meterRegistry,
							this.httpClientCustomizers));

			OpenAIClientAsync resolvedClientAsync = Objects.requireNonNullElseGet(this.openAiClientAsync,
					() -> OpenAiSetup.setupAsyncClient(resolvedOptions.getBaseUrl(), resolvedOptions.getApiKey(),
							resolvedOptions.getCredential(), resolvedOptions.getMicrosoftDeploymentName(),
							resolvedOptions.getMicrosoftFoundryServiceVersion(), resolvedOptions.getOrganizationId(),
							resolvedOptions.isMicrosoftFoundry(), resolvedOptions.isGitHubModels(),
							resolvedOptions.getModel(),
							Objects.requireNonNullElse(resolvedOptions.getTimeout(),
									AbstractOpenAiOptions.DEFAULT_TIMEOUT),
							resolvedOptions.getMaxRetries(), resolvedOptions.getProxy(),
							resolvedOptions.getCustomHeaders(), resolvedObservationRegistry, this.meterRegistry,
							this.httpClientCustomizers));

			ToolCallingManager resolvedToolCallingManager = Objects.requireNonNullElse(this.toolCallingManager,
					ToolCallingManager.builder().observationRegistry(resolvedObservationRegistry).build());

			return new OpenAiChatModel(resolvedClient, resolvedClientAsync, resolvedOptions,
					resolvedObservationRegistry, resolvedToolCallingManager);
		}

	}

}
