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

package org.springframework.ai.openai.responses;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import com.openai.client.OpenAIClient;
import com.openai.client.OpenAIClientAsync;
import com.openai.core.RequestOptions;
import com.openai.core.http.AsyncStreamResponse;
import com.openai.models.responses.Response;
import com.openai.models.responses.ResponseCreateParams;
import com.openai.models.responses.ResponseStatus;
import com.openai.models.responses.ResponseStreamEvent;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationRegistry;
import io.micrometer.observation.contextpropagation.ObservationThreadLocalAccessor;
import org.jspecify.annotations.Nullable;
import reactor.core.publisher.Flux;

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
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.observation.conventions.AiProvider;
import org.springframework.ai.openai.AbstractOpenAiOptions;
import org.springframework.ai.openai.http.okhttp.OpenAiHttpClientBuilderCustomizer;
import org.springframework.ai.openai.setup.OpenAiSetup;
import org.springframework.util.Assert;

/**
 * A {@link ChatModel} backed by OpenAI's Responses API ({@code /v1/responses}), the
 * endpoint where new OpenAI capability lands. Notably, it is the only way to combine
 * reasoning with tool calling on GPT-5.4 and later, which Chat Completions no longer
 * supports.
 * <p>
 * This model is <strong>stateless</strong>. Every call sends the whole {@link Prompt};
 * OpenAI is never asked to remember the conversation, so
 * {@link org.springframework.ai.chat.memory.ChatMemory ChatMemory}, advisors, RAG all
 * keep working exactly as they do with {@code OpenAiChatModel}. The stateful modes are
 * not supported.
 * <p>
 * Like {@code OpenAiChatModel} since 2.0.0, this model runs <em>no</em> tool loop:
 * multi-round tool calling belongs to {@code ToolCallingAdvisor}, which is what keeps the
 * behaviour identical across providers.
 * <p>
 * Known limitations: audio input and output, {@code n}, stop sequences, frequency and
 * presence penalties, {@code seed} and {@code logit_bias} have no equivalent here; many
 * OpenAI-compatible backends implement {@code /v1/chat/completions} but not
 * {@code /v1/responses}; and reasoning continuity across turns depends on the chat memory
 * repository in use, since none of them persists
 * {@link org.springframework.ai.chat.messages.part.MessagePart message parts} yet - see
 * the reference documentation for the details.
 * <p>
 * 中文导读：这是同一 ChatModel 接口的 Responses API 实现，SDK 入口是
 * {@code responses().create/createStreaming}，而非 Chat Completions 的 choices/delta 协议。
 * 请求映射、响应 item 映射和流事件组装分别下沉到 ResponsesRequestBuilder、
 * ResponsesItemMapper、ResponsesStreamAssembler，本类连接它们并管理一次调用的观测生命周期。
 * <p>
 * 请求构建器显式设置 store=false，每轮发送完整 Prompt；本类不依赖服务端保存会话。
 * 函数工具只发送定义并返回调用请求，Java 工具执行循环仍交给 {@code ToolCallingAdvisor}。
 * 因此切换 API 不会把 ChatMemory、RAG 或本地工具执行自动迁到 OpenAI 服务端。
 *
 * @author Dimitar Proynov
 * @since 2.1.0
 * @see org.springframework.ai.openai.OpenAiChatModel
 */
public final class OpenAiResponsesChatModel implements ChatModel {

	private static final ChatModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultChatModelObservationConvention();

	private final OpenAIClient openAiClient;

	private final OpenAIClientAsync openAiClientAsync;

	private final OpenAiResponsesChatOptions options;

	private final ObservationRegistry observationRegistry;

	private final ToolCallingManager toolCallingManager;

	private ChatModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	private OpenAiResponsesChatModel(OpenAIClient openAiClient, OpenAIClientAsync openAiClientAsync,
			OpenAiResponsesChatOptions options, ObservationRegistry observationRegistry,
			ToolCallingManager toolCallingManager) {
		this.openAiClient = openAiClient;
		this.openAiClientAsync = openAiClientAsync;
		this.options = options;
		this.observationRegistry = observationRegistry;
		this.toolCallingManager = toolCallingManager;
	}

	public static Builder builder() {
		return new Builder();
	}

	@Override
	public OpenAiResponsesChatOptions getOptions() {
		return this.options;
	}

	@Override
	public ChatResponse call(Prompt prompt) {
		// 1. 先检查通用选项中不受支持的字段，再按 Responses 的原生/通用选项规则准备 Prompt。
		warnUnsupportedOptions(prompt.getOptions());
		Prompt requestPrompt = buildRequestPrompt(prompt);

		// 2. 消息转为 Responses input items，系统指令和工具定义也在专用 builder 中完成映射。
		ResponseCreateParams request = ResponsesRequestBuilder.build(requestPrompt, this.toolCallingManager);
		RequestOptions requestOptions = buildRequestOptions(requestPrompt);

		ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
			.prompt(requestPrompt)
			.provider(AiProvider.OPENAI.value())
			.build();

		// 3. 同步 SDK 调用与 item 映射共用一次模型 Observation；异常按同步调用契约向外传播。
		return ChatModelObservationDocumentation.CHAT_MODEL_OPERATION
			.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> {
				Response response = this.openAiClient.responses().create(request, requestOptions);
				// HTTP 成功不代表模型响应成功：响应体 status=failed 也需转成调用失败。
				if (isFailed(response)) {
					throw ResponsesItemMapper.failure(response);
				}

				// 多种 output item 组成同一个 Generation 的 message parts，保留文本、推理和工具信息。
				Generation generation = ResponsesItemMapper.toGeneration(response);
				Usage usage = ResponsesItemMapper.toUsage(response);
				ChatResponse chatResponse = new ChatResponse(List.of(generation),
						ResponsesItemMapper.toResponseMetadata(response, usage));

				observationContext.setResponse(chatResponse);
				return chatResponse;
			});
	}

	@Override
	public Flux<ChatResponse> stream(Prompt prompt) {
		warnUnsupportedOptions(prompt.getOptions());
		Prompt requestPrompt = buildRequestPrompt(prompt);

		// 每次订阅重新创建请求、流组装器和 Observation，状态不会跨订阅共享。
		return Flux.deferContextual(contextView -> {
			ResponseCreateParams request = ResponsesRequestBuilder.build(requestPrompt, this.toolCallingManager);
			RequestOptions requestOptions = buildRequestOptions(requestPrompt);

			ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
				.prompt(requestPrompt)
				.provider(AiProvider.OPENAI.value())
				.streaming(true)
				.build();
			Observation observation = ChatModelObservationDocumentation.CHAT_MODEL_OPERATION.observation(
					this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry);
			Observation parentObservation = contextView.getOrDefault(ObservationThreadLocalAccessor.KEY, null);
			// 父子关系来自 Reactor Context；短暂打开父 Scope 仅用于正确创建当前模型的 span。
			observation.parentObservation(parentObservation);
			// Briefly make the parent observation current while starting this one, so
			// Micrometer tracing derives the span's parent from the parent observation
			// rather than from whatever scope happens to be open on the current thread
			// (e.g. the servlet HTTP span).
			try (Observation.Scope ignored = parentObservation != null ? parentObservation.openScope()
					: Observation.Scope.NOOP) {
				observation.start();
			}

			// SDK 按事件回调推送，线程由 SDK/HTTP 客户端决定；本层没有额外调度或重试操作符。
			Flux<ResponseStreamEvent> events = Flux.create(sink -> {
				AsyncStreamResponse<ResponseStreamEvent> response = this.openAiClientAsync.responses()
					.createStreaming(request, requestOptions);
				// 下游取消或流终止时关闭本次 AsyncStreamResponse，释放读取该响应的资源。
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

			// One assembler per subscription: it accumulates the transcript and the
			// running reasoning summary for this stream only.
			ResponsesStreamAssembler assembler = new ResponsesStreamAssembler();
			// 一个协议事件可产生零到多个消息增量；按事件顺序组装，工具参数无需复用 ChatCompletions 的 ChunkMerger。
			Flux<ChatResponse> chatResponses = events.concatMapIterable(assembler::apply);

			// 完成、错误和取消都停止观测；完整回答只供聚合回调记录，仍向调用方透传增量响应。
			Flux<ChatResponse> observedResponses = chatResponses.doOnError(observation::error)
				.doFinally(signal -> observation.stop())
				.contextWrite(ctx -> ctx.put(ObservationThreadLocalAccessor.KEY, observation));

			return new MessageAggregator().aggregate(observedResponses, observationContext::setResponse);
		});
	}

	/**
	 * Use the provided convention for reporting observation data.
	 * @param observationConvention the provided convention
	 */
	public void setObservationConvention(ChatModelObservationConvention observationConvention) {
		Assert.notNull(observationConvention, "observationConvention cannot be null");
		this.observationConvention = observationConvention;
	}

	private static boolean isFailed(Response response) {
		return response.status().filter(status -> ResponseStatus.FAILED.equals(status)).isPresent();
	}

	/**
	 * Warn once per option about portable settings the Responses API has no equivalent
	 * for. Only the incoming options are inspected: once merged into
	 * {@link OpenAiResponsesChatOptions} these read back as {@code null}, and that class
	 * warns from its own builder.
	 */
	private static void warnUnsupportedOptions(@Nullable ChatOptions chatOptions) {
		if (chatOptions == null || chatOptions instanceof OpenAiResponsesChatOptions) {
			return;
		}
		if (chatOptions.getTopK() != null) {
			OpenAiResponsesChatOptions.warnUnsupported("topK");
		}
		if (chatOptions.getStopSequences() != null && !chatOptions.getStopSequences().isEmpty()) {
			OpenAiResponsesChatOptions.warnUnsupported("stopSequences");
		}
		if (chatOptions.getFrequencyPenalty() != null) {
			OpenAiResponsesChatOptions.warnUnsupported("frequencyPenalty");
		}
		if (chatOptions.getPresencePenalty() != null) {
			OpenAiResponsesChatOptions.warnUnsupported("presencePenalty");
		}
	}

	/**
	 * Resolve the options the request is built from.
	 * <p>
	 * A prompt without options gets this model's defaults, and portable
	 * {@link ChatOptions} are merged over them, so a {@code ChatClient} request that only
	 * sets a temperature keeps the configured model, reasoning effort and hosted tools.
	 * <p>
	 * Runtime options that are already {@link OpenAiResponsesChatOptions}
	 * <em>replace</em> the defaults rather than merging with them: they are a complete
	 * specification of the request, so a setting they leave unset falls back to
	 * {@link OpenAiResponsesChatOptions}' own default and not to this model's. This is
	 * what {@code OpenAiChatModel} does with native options too, and it is why
	 * {@link OpenAiResponsesChatOptions#mutate()} exists - start from the model's options
	 * to override one field and keep the rest.
	 */
	private Prompt buildRequestPrompt(Prompt prompt) {
		// 三种情况：无选项用模型默认值；原生选项整体替换；通用 ChatOptions 合并到默认值副本。
		// 这比 OpenAiChatModel 多了通用选项转换分支，不能把两个适配器的参数规则混为一谈。
		ChatOptions runtimeOptions = prompt.getOptions();
		if (runtimeOptions == null) {
			return prompt.mutate().chatOptions(this.options).build();
		}
		if (runtimeOptions instanceof OpenAiResponsesChatOptions) {
			return prompt;
		}
		OpenAiResponsesChatOptions.Builder builder = this.options.mutate();
		builder.combineWith(runtimeOptions.mutate());
		return prompt.mutate().chatOptions(builder.build()).build();
	}

	private RequestOptions buildRequestOptions(Prompt prompt) {
		// 请求级设置只有 timeout；客户端的连接地址、认证和重试在 Builder.build 中确定。
		OpenAiResponsesChatOptions chatOptions = (OpenAiResponsesChatOptions) prompt.getOptions();
		Assert.state(chatOptions != null, "Prompt options must be OpenAiResponsesChatOptions type");
		RequestOptions.Builder requestOptions = RequestOptions.builder();
		if (chatOptions.getTimeout() != null) {
			requestOptions.timeout(chatOptions.getTimeout());
		}
		return requestOptions.build();
	}

	/**
	 * Builder for {@link OpenAiResponsesChatModel} instances. Mirrors
	 * {@code OpenAiChatModel.Builder}, so the two models are configured the same way.
	 */
	public static final class Builder {

		private @Nullable OpenAIClient openAiClient;

		private @Nullable OpenAIClientAsync openAiClientAsync;

		private @Nullable OpenAiResponsesChatOptions options;

		private @Nullable ToolCallingManager toolCallingManager;

		private @Nullable ObservationRegistry observationRegistry;

		private @Nullable MeterRegistry meterRegistry;

		private List<OpenAiHttpClientBuilderCustomizer> httpClientCustomizers = new ArrayList<>();

		private Builder() {
		}

		public Builder openAiClient(OpenAIClient openAiClient) {
			this.openAiClient = openAiClient;
			return this;
		}

		public Builder openAiClientAsync(OpenAIClientAsync openAiClientAsync) {
			this.openAiClientAsync = openAiClientAsync;
			return this;
		}

		public Builder options(OpenAiResponsesChatOptions options) {
			this.options = options;
			return this;
		}

		/**
		 * Sets the tool calling manager used to resolve tool definitions. This model
		 * never executes tools: {@code ToolCallingAdvisor} does.
		 */
		public Builder toolCallingManager(ToolCallingManager toolCallingManager) {
			this.toolCallingManager = toolCallingManager;
			return this;
		}

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
		 * underlying OkHttp client builder before the OpenAI clients are constructed.
		 * Customizers are applied in registration order, after Spring AI's own defaults,
		 * so user code wins.
		 */
		public Builder httpClientBuilderCustomizer(OpenAiHttpClientBuilderCustomizer customizer) {
			Assert.notNull(customizer, "customizer cannot be null");
			this.httpClientCustomizers.add(customizer);
			return this;
		}

		/**
		 * Sets the full list of customizers to apply, replacing any registered earlier on
		 * this builder. The order of the list is preserved.
		 */
		public Builder httpClientBuilderCustomizers(List<OpenAiHttpClientBuilderCustomizer> customizers) {
			Assert.notNull(customizers, "customizers cannot be null");
			this.httpClientCustomizers = new ArrayList<>(customizers);
			return this;
		}

		public OpenAiResponsesChatModel build() {
			// 未显式传客户端才调用 OpenAiSetup；兼容性检查仍会拒绝被识别为 GitHub Models 的配置。
			OpenAiResponsesChatOptions resolvedOptions = Objects.requireNonNullElseGet(this.options,
					() -> OpenAiResponsesChatOptions.builder().build());
			ObservationRegistry resolvedObservationRegistry = Objects.requireNonNullElse(this.observationRegistry,
					ObservationRegistry.NOOP);
			if (OpenAiSetup.detectModelProvider(resolvedOptions.isMicrosoftFoundry(), false,
					resolvedOptions.getBaseUrl(), resolvedOptions.getMicrosoftDeploymentName(),
					resolvedOptions.getMicrosoftFoundryServiceVersion()) == OpenAiSetup.ModelProvider.GITHUB_MODELS) {
				throw new IllegalStateException("GitHub Models does not support the OpenAI Responses API");
			}

			OpenAIClient resolvedClient = Objects.requireNonNullElseGet(this.openAiClient,
					() -> OpenAiSetup.setupSyncClient(resolvedOptions.getBaseUrl(), resolvedOptions.getApiKey(),
							resolvedOptions.getCredential(), resolvedOptions.getMicrosoftDeploymentName(),
							resolvedOptions.getMicrosoftFoundryServiceVersion(), resolvedOptions.getOrganizationId(),
							resolvedOptions.isMicrosoftFoundry(), false, resolvedOptions.getModel(),
							Objects.requireNonNullElse(resolvedOptions.getTimeout(),
									AbstractOpenAiOptions.DEFAULT_TIMEOUT),
							resolvedOptions.getMaxRetries(), resolvedOptions.getProxy(),
							resolvedOptions.getCustomHeaders(), resolvedObservationRegistry, this.meterRegistry,
							this.httpClientCustomizers));

			OpenAIClientAsync resolvedClientAsync = Objects.requireNonNullElseGet(this.openAiClientAsync,
					() -> OpenAiSetup.setupAsyncClient(resolvedOptions.getBaseUrl(), resolvedOptions.getApiKey(),
							resolvedOptions.getCredential(), resolvedOptions.getMicrosoftDeploymentName(),
							resolvedOptions.getMicrosoftFoundryServiceVersion(), resolvedOptions.getOrganizationId(),
							resolvedOptions.isMicrosoftFoundry(), false, resolvedOptions.getModel(),
							Objects.requireNonNullElse(resolvedOptions.getTimeout(),
									AbstractOpenAiOptions.DEFAULT_TIMEOUT),
							resolvedOptions.getMaxRetries(), resolvedOptions.getProxy(),
							resolvedOptions.getCustomHeaders(), resolvedObservationRegistry, this.meterRegistry,
							this.httpClientCustomizers));

			ToolCallingManager resolvedToolCallingManager = Objects.requireNonNullElse(this.toolCallingManager,
					ToolCallingManager.builder().observationRegistry(resolvedObservationRegistry).build());

			return new OpenAiResponsesChatModel(resolvedClient, resolvedClientAsync, resolvedOptions,
					resolvedObservationRegistry, resolvedToolCallingManager);
		}

	}

}
