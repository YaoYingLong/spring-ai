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

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;

import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationRegistry;
import io.micrometer.observation.contextpropagation.ObservationThreadLocalAccessor;
import org.jspecify.annotations.Nullable;
import reactor.core.publisher.Flux;

import org.springframework.ai.chat.client.advisor.ChatModelCallAdvisor;
import org.springframework.ai.chat.client.advisor.ChatModelStreamAdvisor;
import org.springframework.ai.chat.client.advisor.DefaultAroundAdvisorChain;
import org.springframework.ai.chat.client.advisor.StructuredOutputValidationAdvisor;
import org.springframework.ai.chat.client.advisor.ToolCallingAdvisor;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.client.advisor.api.BaseAdvisorChain;
import org.springframework.ai.chat.client.advisor.api.MemoryAdvisor;
import org.springframework.ai.chat.client.advisor.api.ToolAdvisor;
import org.springframework.ai.chat.client.advisor.observation.AdvisorObservationConvention;
import org.springframework.ai.chat.client.observation.ChatClientObservationContext;
import org.springframework.ai.chat.client.observation.ChatClientObservationConvention;
import org.springframework.ai.chat.client.observation.ChatClientObservationDocumentation;
import org.springframework.ai.chat.client.observation.DefaultChatClientObservationConvention;
import org.springframework.ai.chat.messages.AbstractMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.content.Media;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.ai.converter.StructuredOutputConverter;
import org.springframework.ai.support.ToolCallbacks;
import org.springframework.ai.template.TemplateRenderer;
import org.springframework.ai.template.st.StTemplateRenderer;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.Resource;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.MimeType;
import org.springframework.util.StringUtils;

/**
 * The default implementation of {@link ChatClient} as created by the
 * {@link Builder#build()} } method.
 * <p>
 * 中文导读：本类把“构建请求”“选择调用方式”“执行并读取响应”分成三个阶段。
 * 先读 prompt() 与 DefaultChatClientRequestSpec，再读该 Spec 的 call()/stream()，
 * 最后进入 DefaultCallResponseSpec 或 DefaultStreamResponseSpec。
 * 实际组装 Prompt 的代码位于 DefaultChatClientUtils，链的推进位于 DefaultAroundAdvisorChain。
 * <p>
 * 一次同步请求的主线是 prompt().user(...).call().content()：复制默认配置，
 * 收集输入，构建 Advisor 链和 Prompt，启动 Observation，调用链，提取模型文本。
 * entity(...) 还会在调用前传递输出格式/Schema，在调用后执行本地反序列化。
 * <p>
 * 一次客户端请求可能因为工具调用或结构化输出重试产生多次模型请求。
 * 因此客户端、Advisor、模型、工具各层 Observation 的范围不同，排查耗时时要区分层级。
 *
 * @author Mark Pollack
 * @author Christian Tzolov
 * @author Josh Long
 * @author Arjen Poutsma
 * @author Soby Chacko
 * @author Dariusz Jedrzejczyk
 * @author Thomas Vitale
 * @author Jonatan Ivanov
 * @author Wenli Tian
 * @author Sebastien Deleuze
 * @since 1.0.0
 */
public class DefaultChatClient implements ChatClient {

	private static final ChatClientObservationConvention DEFAULT_CHAT_CLIENT_OBSERVATION_CONVENTION = new DefaultChatClientObservationConvention();

	private static final TemplateRenderer DEFAULT_TEMPLATE_RENDERER = StTemplateRenderer.builder().build();

	private static final ChatClientMessageAggregator CHAT_CLIENT_MESSAGE_AGGREGATOR = new ChatClientMessageAggregator();

	private final DefaultChatClientRequestSpec defaultChatClientRequest;

	public DefaultChatClient(DefaultChatClientRequestSpec defaultChatClientRequest) {
		Assert.notNull(defaultChatClientRequest, "defaultChatClientRequest cannot be null");
		this.defaultChatClientRequest = defaultChatClientRequest;
	}

	@Override
	public ChatClientRequestSpec prompt() {
		// 为本次请求复制默认配置；集合容器重新建立，但模型、Advisor、回调等协作对象会复用。
		// RequestSpec 是逐步填写的可变对象，应该在每次业务请求内创建和使用。
		return new DefaultChatClientRequestSpec(this.defaultChatClientRequest);
	}

	@Override
	public ChatClientRequestSpec prompt(String content) {
		Assert.hasText(content, "content cannot be null or empty");
		return prompt(new Prompt(content));
	}

	@Override
	public ChatClientRequestSpec prompt(Prompt prompt) {
		Assert.notNull(prompt, "prompt cannot be null");

		DefaultChatClientRequestSpec spec = new DefaultChatClientRequestSpec(this.defaultChatClientRequest);

		// Messages
		if (prompt.getInstructions() != null) {
			spec.messages(prompt.getInstructions());
		}

		// ChatOptions
		if (prompt.getOptions() != null) {
			ChatOptions.Builder<?> requestOptionsBuilder = prompt.getOptions().mutate();
			if (spec.getOptionsCustomizer() != null) {
				spec.getOptionsCustomizer().combineWith(requestOptionsBuilder);
			}
			else {
				spec.options(requestOptionsBuilder);
			}
		}

		return spec;
	}

	/**
	 * Return a {@link ChatClient.Builder} to create a new {@link ChatClient} whose
	 * settings are replicated from this {@link ChatClientRequest}.
	 */
	@Override
	public Builder mutate() {
		return this.defaultChatClientRequest.mutate();
	}

	public static class DefaultPromptUserSpec implements PromptUserSpec {

		private final Map<String, Object> params = new HashMap<>();

		private final Map<String, Object> metadata = new HashMap<>();

		private final List<Media> media = new ArrayList<>();

		private @Nullable String text;

		@Override
		public PromptUserSpec media(Media... media) {
			Assert.notNull(media, "media cannot be null");
			Assert.noNullElements(media, "media cannot contain null elements");
			this.media.addAll(Arrays.asList(media));
			return this;
		}

		@Override
		public PromptUserSpec media(MimeType mimeType, URL url) {
			Assert.notNull(mimeType, "mimeType cannot be null");
			Assert.notNull(url, "url cannot be null");
			try {
				this.media.add(Media.builder().mimeType(mimeType).data(url.toURI()).build());
			}
			catch (URISyntaxException e) {
				throw new RuntimeException(e);
			}
			return this;
		}

		@Override
		public PromptUserSpec media(MimeType mimeType, Resource resource) {
			Assert.notNull(mimeType, "mimeType cannot be null");
			Assert.notNull(resource, "resource cannot be null");
			this.media.add(Media.builder().mimeType(mimeType).data(resource).build());
			return this;
		}

		@Override
		public PromptUserSpec text(String text) {
			Assert.hasText(text, "text cannot be null or empty");
			this.text = text;
			return this;
		}

		@Override
		public PromptUserSpec text(Resource text, Charset charset) {
			Assert.notNull(text, "text cannot be null");
			Assert.notNull(charset, "charset cannot be null");
			try {
				this.text(text.getContentAsString(charset));
			}
			catch (IOException e) {
				throw new RuntimeException(e);
			}
			return this;
		}

		@Override
		public PromptUserSpec text(Resource text) {
			Assert.notNull(text, "text cannot be null");
			this.text(text, Charset.defaultCharset());
			return this;
		}

		@Override
		public PromptUserSpec param(String key, Object value) {
			Assert.hasText(key, "key cannot be null or empty");
			Assert.notNull(value, "value cannot be null");
			this.params.put(key, value);
			return this;
		}

		@Override
		public PromptUserSpec params(Map<String, Object> params) {
			Assert.notNull(params, "params cannot be null");
			Assert.noNullElements(params.keySet(), "param keys cannot contain null elements");
			Assert.noNullElements(params.values(), "param values cannot contain null elements");
			this.params.putAll(params);
			return this;
		}

		@Override
		public PromptUserSpec metadata(Map<String, Object> metadata) {
			Assert.notNull(metadata, "metadata cannot be null");
			Assert.noNullElements(metadata.keySet(), "metadata keys cannot contain null elements");
			Assert.noNullElements(metadata.values(), "metadata values cannot contain null elements");
			this.metadata.putAll(metadata);
			return this;
		}

		@Override
		public PromptUserSpec metadata(String key, Object value) {
			Assert.hasText(key, "metadata key cannot be null or empty");
			Assert.notNull(value, "metadata value cannot be null");
			this.metadata.put(key, value);
			return this;
		}

		protected @Nullable String text() {
			return this.text;
		}

		protected Map<String, Object> params() {
			return this.params;
		}

		protected List<Media> media() {
			return this.media;
		}

		protected Map<String, Object> metadata() {
			return this.metadata;
		}

	}

	public static class DefaultPromptSystemSpec implements PromptSystemSpec {

		private final Map<String, Object> params = new HashMap<>();

		private final Map<String, Object> metadata = new HashMap<>();

		private @Nullable String text;

		@Override
		public PromptSystemSpec text(String text) {
			Assert.hasText(text, "text cannot be null or empty");
			this.text = text;
			return this;
		}

		@Override
		public PromptSystemSpec text(Resource text, Charset charset) {
			Assert.notNull(text, "text cannot be null");
			Assert.notNull(charset, "charset cannot be null");
			try {
				this.text(text.getContentAsString(charset));
			}
			catch (IOException e) {
				throw new RuntimeException(e);
			}
			return this;
		}

		@Override
		public PromptSystemSpec text(Resource text) {
			Assert.notNull(text, "text cannot be null");
			this.text(text, Charset.defaultCharset());
			return this;
		}

		@Override
		public PromptSystemSpec param(String key, Object value) {
			Assert.hasText(key, "key cannot be null or empty");
			Assert.notNull(value, "value cannot be null");
			this.params.put(key, value);
			return this;
		}

		@Override
		public PromptSystemSpec params(Map<String, Object> params) {
			Assert.notNull(params, "params cannot be null");
			Assert.noNullElements(params.keySet(), "param keys cannot contain null elements");
			Assert.noNullElements(params.values(), "param values cannot contain null elements");
			this.params.putAll(params);
			return this;
		}

		@Override
		public PromptSystemSpec metadata(Map<String, Object> metadata) {
			Assert.notNull(metadata, "metadata cannot be null");
			Assert.noNullElements(metadata.keySet(), "metadata keys cannot contain null elements");
			Assert.noNullElements(metadata.values(), "metadata values cannot contain null elements");
			this.metadata.putAll(metadata);
			return this;
		}

		@Override
		public PromptSystemSpec metadata(String key, Object value) {
			Assert.hasText(key, "metadata key cannot be null or empty");
			Assert.notNull(value, "metadata value cannot be null");
			this.metadata.put(key, value);
			return this;
		}

		protected @Nullable String text() {
			return this.text;
		}

		protected Map<String, Object> params() {
			return this.params;
		}

		protected Map<String, Object> metadata() {
			return this.metadata;
		}

	}

	public static class DefaultAdvisorSpec implements AdvisorSpec {

		private final List<Advisor> advisors = new ArrayList<>();

		private final Map<String, Object> params = new HashMap<>();

		@Override
		public AdvisorSpec param(String key, Object value) {
			Assert.hasText(key, "key cannot be null or empty");
			Assert.notNull(value, "value cannot be null");
			this.params.put(key, value);
			return this;
		}

		@Override
		public AdvisorSpec params(Map<String, Object> params) {
			Assert.notNull(params, "params cannot be null");
			Assert.noNullElements(params.keySet(), "param keys cannot contain null elements");
			Assert.noNullElements(params.values(), "param values cannot contain null elements");
			this.params.putAll(params);
			return this;
		}

		@Override
		public AdvisorSpec advisors(Advisor... advisors) {
			Assert.notNull(advisors, "advisors cannot be null");
			Assert.noNullElements(advisors, "advisors cannot contain null elements");
			this.advisors.addAll(List.of(advisors));
			return this;
		}

		@Override
		public AdvisorSpec advisors(List<Advisor> advisors) {
			Assert.notNull(advisors, "advisors cannot be null");
			Assert.noNullElements(advisors, "advisors cannot contain null elements");
			this.advisors.addAll(advisors);
			return this;
		}

		public List<Advisor> getAdvisors() {
			return this.advisors;
		}

		public Map<String, Object> getParams() {
			return this.params;
		}

	}

	/**
	 * Default implementation of {@link EntityParamSpec}.
	 */
	public static class DefaultEntityParamSpec implements EntityParamSpec {

		private boolean enableNative = false;

		private boolean validated = false;

		public boolean isEnableNative() {
			return this.enableNative;
		}

		public boolean isValidated() {
			return this.validated;
		}

		@Override
		public EntityParamSpec useProviderStructuredOutput() {
			this.enableNative = true;
			return this;
		}

		@Override
		public EntityParamSpec validateSchema() {
			this.validated = true;
			return this;
		}

	}

	/**
	 * 同步结果入口：持有已构建的请求与执行链，但构造本对象不会调用模型。
	 * content()/chatResponse()/entity() 等读取操作才向下执行，并不是对缓存结果的不同视图。
	 * 链会弹出已执行的 Advisor，因此一次请求应选择一个结果读取入口；同时需要原始响应和
	 * 反序列化对象时，使用 responseEntity(...)。
	 */
	public static class DefaultCallResponseSpec implements CallResponseSpec {

		private final ChatClientRequest request;

		private final BaseAdvisorChain advisorChain;

		private final ObservationRegistry observationRegistry;

		private final ChatClientObservationConvention observationConvention;

		public DefaultCallResponseSpec(ChatClientRequest chatClientRequest, BaseAdvisorChain advisorChain,
				ObservationRegistry observationRegistry, ChatClientObservationConvention observationConvention) {
			Assert.notNull(chatClientRequest, "chatClientRequest cannot be null");
			Assert.notNull(advisorChain, "advisorChain cannot be null");
			Assert.notNull(observationRegistry, "observationRegistry cannot be null");
			Assert.notNull(observationConvention, "observationConvention cannot be null");

			this.request = chatClientRequest;
			this.advisorChain = advisorChain;
			this.observationRegistry = observationRegistry;
			this.observationConvention = observationConvention;
		}

		@Override
		public <T> ResponseEntity<ChatResponse, T> responseEntity(Class<T> type,
				Consumer<EntityParamSpec> entityParamSpecConsumer) {
			Assert.notNull(type, "type cannot be null");
			Assert.notNull(entityParamSpecConsumer, "entityParamSpecConsumer cannot be null");
			var converter = new BeanOutputConverter<>(type);
			return doResponseEntity(converter, resolveAdvisorChain(entityParamSpecConsumer, converter));
		}

		@Override
		public <T> ResponseEntity<ChatResponse, T> responseEntity(Class<T> type) {
			Assert.notNull(type, "type cannot be null");
			return doResponseEntity(new BeanOutputConverter<>(type));
		}

		@Override
		public <T> ResponseEntity<ChatResponse, T> responseEntity(ParameterizedTypeReference<T> type,
				Consumer<EntityParamSpec> entityParamSpecConsumer) {
			Assert.notNull(type, "type cannot be null");
			Assert.notNull(entityParamSpecConsumer, "entityParamSpecConsumer cannot be null");
			var converter = new BeanOutputConverter<>(type);
			return doResponseEntity(converter, resolveAdvisorChain(entityParamSpecConsumer, converter));
		}

		@Override
		public <T> ResponseEntity<ChatResponse, T> responseEntity(ParameterizedTypeReference<T> type) {
			Assert.notNull(type, "type cannot be null");
			return doResponseEntity(new BeanOutputConverter<>(type));
		}

		@Override
		public <T> ResponseEntity<ChatResponse, T> responseEntity(
				StructuredOutputConverter<T> structuredOutputConverter,
				Consumer<EntityParamSpec> entityParamSpecConsumer) {
			Assert.notNull(structuredOutputConverter, "structuredOutputConverter cannot be null");
			Assert.notNull(entityParamSpecConsumer, "entityParamSpecConsumer cannot be null");
			return doResponseEntity(structuredOutputConverter,
					resolveAdvisorChain(entityParamSpecConsumer, structuredOutputConverter));
		}

		@Override
		public <T> ResponseEntity<ChatResponse, T> responseEntity(
				StructuredOutputConverter<T> structuredOutputConverter) {
			Assert.notNull(structuredOutputConverter, "structuredOutputConverter cannot be null");
			return doResponseEntity(structuredOutputConverter);
		}

		protected <T> ResponseEntity<ChatResponse, T> doResponseEntity(StructuredOutputConverter<T> outputConverter) {
			return this.doResponseEntity(outputConverter, this.advisorChain);
		}

		protected <T> ResponseEntity<ChatResponse, T> doResponseEntity(StructuredOutputConverter<T> outputConverter,
				BaseAdvisorChain advisorChain) {

			Assert.notNull(outputConverter, "structuredOutputConverter cannot be null");
			Assert.notNull(advisorChain, "advisor chain cannot be null");

			this.request.context().put(ChatClientAttributes.OUTPUT_FORMAT.getKey(), outputConverter.getFormat());

			if (Boolean.TRUE
				.equals(this.request.context().get(ChatClientAttributes.STRUCTURED_OUTPUT_NATIVE.getKey()))) {
				this.request.context()
					.put(ChatClientAttributes.STRUCTURED_OUTPUT_SCHEMA.getKey(), outputConverter.getJsonSchema());
			}

			var chatResponse = doGetObservableChatClientResponse(this.request, advisorChain).chatResponse();
			var responseContent = getContentFromChatResponse(chatResponse);
			if (responseContent == null) {
				return new ResponseEntity<>(chatResponse, null);
			}
			T entity = outputConverter.convert(responseContent);
			return new ResponseEntity<>(chatResponse, entity);
		}

		@Override
		public <T> @Nullable T entity(ParameterizedTypeReference<T> type,
				Consumer<EntityParamSpec> entitySpecConsumer) {
			Assert.notNull(type, "type cannot be null");
			Assert.notNull(entitySpecConsumer, "entitySpecConsumer cannot be null");
			var converter = new BeanOutputConverter<>(type);
			return doSingleWithBeanOutputConverter(converter, resolveAdvisorChain(entitySpecConsumer, converter));
		}

		@Override
		public <T> @Nullable T entity(Class<T> type, Consumer<EntityParamSpec> entitySpecConsumer) {
			Assert.notNull(type, "type cannot be null");
			Assert.notNull(entitySpecConsumer, "entitySpecConsumer cannot be null");
			var converter = new BeanOutputConverter<>(type);
			return doSingleWithBeanOutputConverter(converter, resolveAdvisorChain(entitySpecConsumer, converter));
		}

		@Override
		public <T> @Nullable T entity(ParameterizedTypeReference<T> type) {
			Assert.notNull(type, "type cannot be null");
			return doSingleWithBeanOutputConverter(new BeanOutputConverter<>(type));
		}

		@Override
		public <T> @Nullable T entity(StructuredOutputConverter<T> structuredOutputConverter,
				Consumer<EntityParamSpec> entitySpecConsumer) {
			Assert.notNull(structuredOutputConverter, "structuredOutputConverter cannot be null");
			Assert.notNull(entitySpecConsumer, "entitySpecConsumer cannot be null");
			return doSingleWithBeanOutputConverter(structuredOutputConverter,
					resolveAdvisorChain(entitySpecConsumer, structuredOutputConverter));
		}

		@Override
		public <T> @Nullable T entity(StructuredOutputConverter<T> structuredOutputConverter) {
			Assert.notNull(structuredOutputConverter, "structuredOutputConverter cannot be null");
			return doSingleWithBeanOutputConverter(structuredOutputConverter);
		}

		@Override
		public <T> @Nullable T entity(Class<T> type) {
			Assert.notNull(type, "type cannot be null");
			return doSingleWithBeanOutputConverter(new BeanOutputConverter<>(type));
		}

		private BaseAdvisorChain resolveAdvisorChain(Consumer<EntityParamSpec> consumer,
				StructuredOutputConverter<?> converter) {
			// 两个独立选项：native 控制 Schema 如何发送，validated 控制是否在本地验证并反馈重试。
			// 仅选择 entity(Class) 并不等于自动启用 Schema 验证 Advisor。
			var spec = new DefaultEntityParamSpec();
			consumer.accept(spec);
			if (spec.isEnableNative()) {
				this.request.context().put(ChatClientAttributes.STRUCTURED_OUTPUT_NATIVE.getKey(), true);
			}
			if (spec.isValidated()) {
				var validationAdvisor = StructuredOutputValidationAdvisor.builder()
					.outputJsonSchema(converter.getJsonSchema())
					.build();
				return this.advisorChain.mutate().push(validationAdvisor).build();
			}
			return this.advisorChain;
		}

		private <T> @Nullable T doSingleWithBeanOutputConverter(StructuredOutputConverter<T> outputConverter) {
			return doSingleWithBeanOutputConverter(outputConverter, this.advisorChain);
		}

		private <T> @Nullable T doSingleWithBeanOutputConverter(StructuredOutputConverter<T> outputConverter,
				BaseAdvisorChain advisorChain) {

			// 转换器同时承担“告诉模型应该输出什么”和“把响应文本转成 Java 对象”两个方向。
			// 先把格式写入 context，链末尾的 ChatModelCallAdvisor 再落实到 Prompt 或模型选项。
			if (StringUtils.hasText(outputConverter.getFormat())) {
				// Used for default structured output format support, based on prompt
				// instructions.
				this.request.context().put(ChatClientAttributes.OUTPUT_FORMAT.getKey(), outputConverter.getFormat());
			}

			if (Boolean.TRUE
				.equals(this.request.context().get(ChatClientAttributes.STRUCTURED_OUTPUT_NATIVE.getKey()))) {
				// Used for native structured output support, e.g. AI model API should
				// provide structured output support.
				this.request.context()
					.put(ChatClientAttributes.STRUCTURED_OUTPUT_SCHEMA.getKey(), outputConverter.getJsonSchema());

			}

			var chatResponse = doGetObservableChatClientResponse(this.request, advisorChain).chatResponse();

			var stringResponse = getContentFromChatResponse(chatResponse);
			if (stringResponse == null) {
				return null;
			}
			// 本地反序列化发生在模型返回后；格式错误会按转换器契约抛出异常，不会自动重发请求。
			return outputConverter.convert(stringResponse);
		}

		@Override
		public ChatClientResponse chatClientResponse() {
			return doGetObservableChatClientResponse(this.request);
		}

		@Override
		public @Nullable ChatResponse chatResponse() {
			return doGetObservableChatClientResponse(this.request).chatResponse();
		}

		@Override
		public @Nullable String content() {
			ChatResponse chatResponse = doGetObservableChatClientResponse(this.request).chatResponse();
			return getContentFromChatResponse(chatResponse);
		}

		private ChatClientResponse doGetObservableChatClientResponse(ChatClientRequest chatClientRequest) {
			return doGetObservableChatClientResponse(chatClientRequest, this.advisorChain);
		}

		private ChatClientResponse doGetObservableChatClientResponse(ChatClientRequest chatClientRequest,
				BaseAdvisorChain advisorChain) {

			// 客户端观测包围整个 Advisor 链，因此包含检索、工具循环等开销。
			// 模型适配器内部还有单次模型请求的观测，两者不是重复统计同一层职责。
			String outputFormat = (String) chatClientRequest.context()
				.getOrDefault(ChatClientAttributes.OUTPUT_FORMAT.getKey(), null);

			ChatClientObservationContext observationContext = ChatClientObservationContext.builder()
				.request(chatClientRequest)
				.advisors(advisorChain.getCallAdvisors())
				.stream(false)
				.format(outputFormat)
				.build();

			var observation = ChatClientObservationDocumentation.AI_CHAT_CLIENT.observation(this.observationConvention,
					DEFAULT_CHAT_CLIENT_OBSERVATION_CONVENTION, () -> observationContext, this.observationRegistry);

			// CHECKSTYLE:OFF
			var chatClientResponse = observation.observe(() -> {
				// Apply the advisor chain that terminates with the ChatModelCallAdvisor.
				var response = advisorChain.nextCall(chatClientRequest);
				observationContext.setResponse(response);
				return response;
			});
			// CHECKSTYLE:ON
			return chatClientResponse != null ? chatClientResponse : ChatClientResponse.builder().build();
		}

		private static @Nullable String getContentFromChatResponse(@Nullable ChatResponse chatResponse) {
			return Optional.ofNullable(chatResponse)
				.map(ChatResponse::getResult)
				.map(Generation::getOutput)
				.map(AbstractMessage::getText)
				.orElse(null);
		}

	}

	/**
	 * 流式结果入口：构建 Flux 描述后续处理，订阅时才进入 Advisor 链。
	 * 文本片段会逐个传给调用方，同时在旁路累积完整响应供观测使用。
	 * 当前响应 Spec 持有会被消费的链，不应把同一个实例当成可反复订阅的请求工厂。
	 */
	public static class DefaultStreamResponseSpec implements StreamResponseSpec {

		private final ChatClientRequest request;

		private final BaseAdvisorChain advisorChain;

		private final ObservationRegistry observationRegistry;

		private final ChatClientObservationConvention observationConvention;

		public DefaultStreamResponseSpec(ChatClientRequest chatClientRequest, BaseAdvisorChain advisorChain,
				ObservationRegistry observationRegistry, ChatClientObservationConvention observationConvention) {
			Assert.notNull(chatClientRequest, "chatClientRequest cannot be null");
			Assert.notNull(advisorChain, "advisorChain cannot be null");
			Assert.notNull(observationRegistry, "observationRegistry cannot be null");
			Assert.notNull(observationConvention, "observationConvention cannot be null");

			this.request = chatClientRequest;
			this.advisorChain = advisorChain;
			this.observationRegistry = observationRegistry;
			this.observationConvention = observationConvention;
		}

		private Flux<ChatClientResponse> doGetObservableFluxChatResponse(ChatClientRequest chatClientRequest) {
			// 从订阅者的 Reactor Context 获取父 Observation，不能只依赖当前线程的 ThreadLocal。
			// 流式信号可能经过多个调度器，观测上下文需要沿订阅链传播。
			return Flux.deferContextual(contextView -> {

				ChatClientObservationContext observationContext = ChatClientObservationContext.builder()
					.request(chatClientRequest)
					.advisors(this.advisorChain.getStreamAdvisors())
					.stream(true)
					.build();

				Observation observation = ChatClientObservationDocumentation.AI_CHAT_CLIENT.observation(
						this.observationConvention, DEFAULT_CHAT_CLIENT_OBSERVATION_CONVENTION,
						() -> observationContext, this.observationRegistry);

				Observation parentObservation = contextView.getOrDefault(ObservationThreadLocalAccessor.KEY, null);
				observation.parentObservation(parentObservation);
				// Briefly make the parent observation current while starting this one, so
				// Micrometer tracing derives the span's parent from the parent
				// observation rather than from whatever scope happens to be open on the
				// current thread (e.g. the servlet HTTP span). This keeps span parenting
				// correct without relying on automatic context propagation.
				try (Observation.Scope ignored = parentObservation != null ? parentObservation.openScope()
						: Observation.Scope.NOOP) {
					observation.start();
				}

				// doFinally 覆盖正常完成、异常和取消，避免只在完成时结束观测造成资源生命周期失配。
				// @formatter:off
				// Apply the advisor chain that terminates with the ChatModelStreamAdvisor.
				Flux<ChatClientResponse> chatClientResponse = this.advisorChain.nextStream(chatClientRequest)
						.doOnError(observation::error)
						.doFinally(s -> observation.stop())
						.contextWrite(ctx -> ctx.put(ObservationThreadLocalAccessor.KEY, observation));
				// @formatter:on
				// 聚合器保留原始增量流，只在正常完成时把聚合结果交给观测回调。
				return CHAT_CLIENT_MESSAGE_AGGREGATOR.aggregateChatClientResponse(chatClientResponse,
						observationContext::setResponse);
			});
		}

		@Override
		public Flux<ChatClientResponse> chatClientResponse() {
			return doGetObservableFluxChatResponse(this.request);
		}

		@Override
		@SuppressWarnings("NullAway") // https://github.com/uber/NullAway/issues/1290
		public Flux<ChatResponse> chatResponse() {
			return doGetObservableFluxChatResponse(this.request).mapNotNull(ChatClientResponse::chatResponse);
		}

		@Override
		public Flux<String> content() {
			// @formatter:off
			return chatResponse()
					.map(r -> Optional.ofNullable(r.getResult())
							.map(Generation::getOutput)
							.map(AbstractMessage::getText)
							.orElse(""))
					.filter(StringUtils::hasLength);
			// @formatter:on
		}

	}

	/**
	 * 单次请求的配置收集器。user/system 的模板变量用于渲染文本；advisorParams 用于
	 * Advisor 间传参；toolContext 用于工具执行。这三种 Map 不会自动互相转换。
	 * setters 主要保存配置，转换成 Prompt、模型选项和请求 context 的步骤统一放在
	 * DefaultChatClientUtils.toChatClientRequest 中。
	 */
	public static class DefaultChatClientRequestSpec implements ChatClientRequestSpec {

		private final ObservationRegistry observationRegistry;

		private final ChatClientObservationConvention chatClientObservationConvention;

		private final @Nullable AdvisorObservationConvention advisorObservationConvention;

		private final ChatModel chatModel;

		private final List<Media> media = new ArrayList<>();

		private final List<ToolCallback> toolCallbacks = new ArrayList<>();

		private final List<ToolCallbackProvider> toolCallbackProviders = new ArrayList<>();

		private final List<Message> messages = new ArrayList<>();

		private final Map<String, Object> userParams = new HashMap<>();

		private final Map<String, Object> userMetadata = new HashMap<>();

		private final Map<String, Object> systemParams = new HashMap<>();

		private final Map<String, Object> systemMetadata = new HashMap<>();

		private final List<Advisor> advisors = new ArrayList<>();

		private final Map<String, Object> advisorParams = new HashMap<>();

		private final Map<String, Object> toolContext = new HashMap<>();

		private TemplateRenderer templateRenderer;

		private @Nullable String userText;

		private @Nullable String systemText;

		private ChatOptions.@Nullable Builder<?> optionsCustomizer;

		private final ToolCallingAdvisor.Builder<?> toolCallingAdvisorBuilder;

		/* copy constructor */
		DefaultChatClientRequestSpec(DefaultChatClientRequestSpec ccr) {
			this(ccr.chatModel, ccr.userText, ccr.userParams, ccr.userMetadata, ccr.systemText, ccr.systemParams,
					ccr.systemMetadata, ccr.toolCallbacks, ccr.toolCallbackProviders, ccr.messages, ccr.media,
					ccr.optionsCustomizer, ccr.advisors, ccr.advisorParams, ccr.observationRegistry,
					ccr.chatClientObservationConvention, ccr.toolContext, ccr.templateRenderer,
					ccr.advisorObservationConvention, ccr.toolCallingAdvisorBuilder);
		}

		public DefaultChatClientRequestSpec(ChatModel chatModel, @Nullable String userText,
				Map<String, Object> userParams, Map<String, Object> userMetadata, @Nullable String systemText,
				Map<String, Object> systemParams, Map<String, Object> systemMetadata, List<ToolCallback> toolCallbacks,
				List<ToolCallbackProvider> toolCallbackProviders, List<Message> messages, List<Media> media,
				ChatOptions.@Nullable Builder<?> customizer, List<Advisor> advisors, Map<String, Object> advisorParams,
				ObservationRegistry observationRegistry,
				@Nullable ChatClientObservationConvention chatClientObservationConvention,
				Map<String, Object> toolContext, @Nullable TemplateRenderer templateRenderer,
				@Nullable AdvisorObservationConvention advisorObservationConvention,
				ToolCallingAdvisor.Builder<?> toolCallingAdvisorBuilder) {

			Assert.notNull(chatModel, "chatModel cannot be null");
			Assert.notNull(userParams, "userParams cannot be null");
			Assert.notNull(userMetadata, "userMetadata cannot be null");
			Assert.notNull(systemParams, "systemParams cannot be null");
			Assert.notNull(systemMetadata, "systemMetadata cannot be null");
			Assert.notNull(toolCallbacks, "toolCallbacks cannot be null");
			Assert.notNull(toolCallbackProviders, "toolCallbackProviders cannot be null");
			Assert.notNull(messages, "messages cannot be null");
			Assert.notNull(media, "media cannot be null");
			Assert.notNull(advisors, "advisors cannot be null");
			Assert.notNull(advisorParams, "advisorParams cannot be null");
			Assert.notNull(observationRegistry, "observationRegistry cannot be null");
			Assert.notNull(toolContext, "toolContext cannot be null");
			Assert.notNull(toolCallingAdvisorBuilder, "toolCallingAdvisorBuilder cannot be null");

			this.chatModel = chatModel;
			this.toolCallingAdvisorBuilder = toolCallingAdvisorBuilder;
			// 克隆选项 Builder，并复制下面的 List/Map 容器，隔离不同请求对配置集合的修改。
			// 这不是整个对象图的深拷贝；例如 Advisor 实例仍可由多个请求共享。
			this.optionsCustomizer = customizer != null ? customizer.clone() : null;

			this.userText = userText;
			this.userParams.putAll(userParams);
			this.userMetadata.putAll(userMetadata);

			this.systemText = systemText;
			this.systemParams.putAll(systemParams);
			this.systemMetadata.putAll(systemMetadata);

			this.toolCallbacks.addAll(toolCallbacks);
			this.toolCallbackProviders.addAll(toolCallbackProviders);
			this.messages.addAll(messages);
			this.media.addAll(media);
			this.advisors.addAll(advisors);
			this.advisorParams.putAll(advisorParams);
			this.observationRegistry = observationRegistry;
			this.chatClientObservationConvention = chatClientObservationConvention != null
					? chatClientObservationConvention : DEFAULT_CHAT_CLIENT_OBSERVATION_CONVENTION;
			this.toolContext.putAll(toolContext);
			this.templateRenderer = templateRenderer != null ? templateRenderer : DEFAULT_TEMPLATE_RENDERER;
			this.advisorObservationConvention = advisorObservationConvention;
		}

		public @Nullable String getUserText() {
			return this.userText;
		}

		public Map<String, Object> getUserParams() {
			return this.userParams;
		}

		public Map<String, Object> getUserMetadata() {
			return this.userMetadata;
		}

		public @Nullable String getSystemText() {
			return this.systemText;
		}

		public Map<String, Object> getSystemParams() {
			return this.systemParams;
		}

		public Map<String, Object> getSystemMetadata() {
			return this.systemMetadata;
		}

		public List<Advisor> getAdvisors() {
			return this.advisors;
		}

		public Map<String, Object> getAdvisorParams() {
			return this.advisorParams;
		}

		public List<Message> getMessages() {
			return this.messages;
		}

		public List<Media> getMedia() {
			return this.media;
		}

		public List<ToolCallback> getToolCallbacks() {
			return this.toolCallbacks;
		}

		public List<ToolCallbackProvider> getToolCallbackProviders() {
			return this.toolCallbackProviders;
		}

		public Map<String, Object> getToolContext() {
			return this.toolContext;
		}

		public TemplateRenderer getTemplateRenderer() {
			return this.templateRenderer;
		}

		/* package */ ChatModel getChatModel() {
			return this.chatModel;
		}

		/* package */ ChatOptions.@Nullable Builder<?> getOptionsCustomizer() {
			return this.optionsCustomizer;
		}

		/**
		 * Return a {@link ChatClient.Builder} to create a new {@link ChatClient} whose
		 * settings are replicated from this {@link ChatClientRequest}.
		 */
		@Override
		public Builder mutate() {
			DefaultChatClientBuilder builder = (DefaultChatClientBuilder) ChatClient
				.builder(this.chatModel, this.observationRegistry, this.chatClientObservationConvention,
						this.advisorObservationConvention, this.toolCallingAdvisorBuilder)
				.defaultTemplateRenderer(this.templateRenderer)
				.defaultTools(this.toolCallbacks.toArray(new ToolCallback[0]))
				.defaultTools((Object[]) this.toolCallbackProviders.toArray(new ToolCallbackProvider[0]))
				.defaultToolContext(this.toolContext);

			if (!CollectionUtils.isEmpty(this.advisors)) {
				builder.defaultAdvisors(a -> a.advisors(this.advisors).params(this.advisorParams));
			}

			if (StringUtils.hasText(this.userText)) {
				String text = this.userText;
				builder.defaultUser(u -> u.text(text)
					.params(this.userParams)
					.media(this.media.toArray(new Media[0]))
					.metadata(this.userMetadata));
			}

			if (StringUtils.hasText(this.systemText)) {
				String text = this.systemText;
				builder.defaultSystem(s -> s.text(text).params(this.systemParams).metadata(this.systemMetadata));
			}

			if (this.optionsCustomizer != null) {
				builder.defaultOptions(this.optionsCustomizer);
			}

			builder.addMessages(this.messages);

			return builder;
		}

		@Override
		public ChatClientRequestSpec advisors(Consumer<ChatClient.AdvisorSpec> consumer) {
			Assert.notNull(consumer, "consumer cannot be null");
			var advisorSpec = new DefaultAdvisorSpec();
			consumer.accept(advisorSpec);
			this.advisorParams.putAll(advisorSpec.getParams());
			this.advisors.addAll(advisorSpec.getAdvisors());
			return this;
		}

		@Override
		public ChatClientRequestSpec advisors(Advisor... advisors) {
			Assert.notNull(advisors, "advisors cannot be null");
			Assert.noNullElements(advisors, "advisors cannot contain null elements");
			this.advisors.addAll(Arrays.asList(advisors));
			return this;
		}

		@Override
		public ChatClientRequestSpec advisors(List<Advisor> advisors) {
			Assert.notNull(advisors, "advisors cannot be null");
			Assert.noNullElements(advisors, "advisors cannot contain null elements");
			this.advisors.addAll(advisors);
			return this;
		}

		@Override
		public ChatClientRequestSpec messages(Message... messages) {
			Assert.notNull(messages, "messages cannot be null");
			Assert.noNullElements(messages, "messages cannot contain null elements");
			this.messages.addAll(List.of(messages));
			return this;
		}

		@Override
		public ChatClientRequestSpec messages(List<Message> messages) {
			Assert.notNull(messages, "messages cannot be null");
			Assert.noNullElements(messages, "messages cannot contain null elements");
			this.messages.addAll(messages);
			return this;
		}

		@Override
		public <B extends ChatOptions.Builder<?>> ChatClientRequestSpec options(B customizer) {
			Assert.notNull(customizer, "customizer cannot be null");
			this.optionsCustomizer = customizer;
			return this;
		}

		@Override
		public ChatClientRequestSpec toolCallbacks(ToolCallback... toolCallbacks) {
			return tools(toolCallbacks);
		}

		@Override
		public ChatClientRequestSpec toolCallbacks(List<ToolCallback> toolCallbacks) {
			return tools(toolCallbacks);
		}

		@Override
		public ChatClientRequestSpec tools(Object... toolObjects) {
			Assert.notNull(toolObjects, "toolObjects cannot be null");
			Assert.noNullElements(toolObjects, "toolObjects cannot contain null elements");

			// 统一工具注册入口：已有回调直接保存，Provider 延迟展开，普通对象转为 @Tool 方法回调。
			// 注册仅提供模型可选择的工具定义与本地执行入口，这里还没有执行用户的方法。
			List<Object> pojos = new ArrayList<>();
			for (Object toolObject : toolObjects) {
				if (toolObject instanceof ToolCallback toolCallback) {
					this.toolCallbacks.add(toolCallback);
				}
				else if (toolObject instanceof ToolCallbackProvider toolCallbackProvider) {
					this.toolCallbackProviders.add(toolCallbackProvider);
				}
				else if (toolObject instanceof ToolCallback[] callbacks) {
					Assert.noNullElements(callbacks, "toolCallbacks cannot contain null elements");
					this.toolCallbacks.addAll(Arrays.asList(callbacks));
				}
				else if (toolObject instanceof ToolCallbackProvider[] providers) {
					Assert.noNullElements(providers, "toolCallbackProviders cannot contain null elements");
					this.toolCallbackProviders.addAll(Arrays.asList(providers));
				}
				else if (toolObject instanceof Collection<?> collection) {
					Assert.noNullElements(collection, "toolObjects collection cannot contain null elements");
					for (Object element : collection) {
						if (element instanceof ToolCallback toolCallback) {
							this.toolCallbacks.add(toolCallback);
						}
						else if (element instanceof ToolCallbackProvider toolCallbackProvider) {
							this.toolCallbackProviders.add(toolCallbackProvider);
						}
						else {
							pojos.add(element);
						}
					}
				}
				else {
					pojos.add(toolObject);
				}
			}

			if (!pojos.isEmpty()) {
				this.toolCallbacks.addAll(Arrays.asList(ToolCallbacks.from(pojos.toArray())));
			}

			return this;
		}

		@Override
		public ChatClientRequestSpec toolCallbacks(ToolCallbackProvider... toolCallbackProviders) {
			return tools(toolCallbackProviders);
		}

		@Override
		public ChatClientRequestSpec toolContext(Map<String, Object> toolContext) {
			Assert.notNull(toolContext, "context cannot be null");
			Assert.noNullElements(toolContext.keySet(), "context keys cannot contain null elements");
			Assert.noNullElements(toolContext.values(), "context values cannot contain null elements");
			toolContext.keySet().forEach(key -> Assert.hasText(key, "context key cannot be null or empty"));
			this.toolContext.putAll(toolContext);
			return this;
		}

		@Override
		public ChatClientRequestSpec system(String text) {
			Assert.hasText(text, "text cannot be null or empty");
			this.systemText = text;
			return this;
		}

		@Override
		public ChatClientRequestSpec system(Resource text, Charset charset) {
			Assert.notNull(text, "text cannot be null");
			Assert.notNull(charset, "charset cannot be null");

			try {
				this.systemText = text.getContentAsString(charset);
			}
			catch (IOException e) {
				throw new RuntimeException(e);
			}
			return this;
		}

		@Override
		public ChatClientRequestSpec system(Resource text) {
			Assert.notNull(text, "text cannot be null");
			return this.system(text, Charset.defaultCharset());
		}

		@Override
		public ChatClientRequestSpec system(Consumer<PromptSystemSpec> consumer) {
			Assert.notNull(consumer, "consumer cannot be null");

			var systemSpec = new DefaultPromptSystemSpec();
			consumer.accept(systemSpec);
			this.systemText = StringUtils.hasText(systemSpec.text()) ? systemSpec.text() : this.systemText;
			this.systemParams.putAll(systemSpec.params());
			this.systemMetadata.putAll(systemSpec.metadata());
			return this;
		}

		@Override
		public ChatClientRequestSpec user(String text) {
			Assert.hasText(text, "text cannot be null or empty");
			this.userText = text;
			return this;
		}

		@Override
		public ChatClientRequestSpec user(Resource text, Charset charset) {
			Assert.notNull(text, "text cannot be null");
			Assert.notNull(charset, "charset cannot be null");

			try {
				this.userText = text.getContentAsString(charset);
			}
			catch (IOException e) {
				throw new RuntimeException(e);
			}
			return this;
		}

		@Override
		public ChatClientRequestSpec user(Resource text) {
			Assert.notNull(text, "text cannot be null");
			return this.user(text, Charset.defaultCharset());
		}

		@Override
		public ChatClientRequestSpec user(Consumer<PromptUserSpec> consumer) {
			Assert.notNull(consumer, "consumer cannot be null");

			// user(u -> u.text(...).param(...)) 的 lambda 在这里立即执行，u 就是下面的 us。
			// lambda 收集的是文本和变量；真正模板替换发生在 toChatClientRequest 中。
			var us = new DefaultPromptUserSpec();
			consumer.accept(us);
			this.userText = StringUtils.hasText(us.text()) ? us.text() : this.userText;
			this.userParams.putAll(us.params());
			this.media.addAll(us.media());
			this.userMetadata.putAll(us.metadata());
			return this;
		}

		@Override
		public ChatClientRequestSpec templateRenderer(TemplateRenderer templateRenderer) {
			Assert.notNull(templateRenderer, "templateRenderer cannot be null");
			this.templateRenderer = templateRenderer;
			return this;
		}

		@Override
		public CallResponseSpec call() {
			// 构建执行链并冻结本轮输入到请求对象；远程调用在响应 Spec 的结果读取方法中发生。
			BaseAdvisorChain advisorChain = buildAdvisorChain();
			return new DefaultCallResponseSpec(DefaultChatClientUtils.toChatClientRequest(this), advisorChain,
					this.observationRegistry, this.chatClientObservationConvention);
		}

		@Override
		public StreamResponseSpec stream() {
			// 同样先构建配置；返回流式响应入口后，Flux 的订阅才驱动链执行。
			BaseAdvisorChain advisorChain = buildAdvisorChain();
			return new DefaultStreamResponseSpec(DefaultChatClientUtils.toChatClientRequest(this), advisorChain,
					this.observationRegistry, this.chatClientObservationConvention);
		}

		private BaseAdvisorChain buildAdvisorChain() {
			// 工具循环放在 Advisor 层：先补齐默认工具 Advisor，再确保整条链只有一个工具调度者。
			autoRegisterToolCallingAdvisor();
			validateSingleToolAdvisor();

			// At the stack bottom add the model call advisors.
			// They play the role of the last advisors in the advisor chain.
			// 同步和流式终点都加入候选集合，DefaultAroundAdvisorChain 会按接口类型拆成两条链。
			// 实际顺序由 getOrder() 决定，数值越小越早处理请求；响应沿嵌套调用方向返回。
			List<Advisor> chain = new ArrayList<>(this.advisors);
			chain.add(ChatModelCallAdvisor.builder().chatModel(this.chatModel).build());
			chain.add(ChatModelStreamAdvisor.builder().chatModel(this.chatModel).build());

			return DefaultAroundAdvisorChain.builder(this.observationRegistry)
				.observationConvention(this.advisorObservationConvention)
				.pushAll(chain)
				.build();
		}

		/**
		 * Auto-registers a {@link ToolCallingAdvisor} unless auto-registration is
		 * disabled or a {@link ToolAdvisor} is already present in the chain. The advisor
		 * is always registered so that tools injected dynamically at runtime (e.g. by
		 * another advisor) are handled correctly even when no static tools are configured
		 * on the call.
		 * <p>
		 * Disables the advisor's internal conversation history when a
		 * {@link MemoryAdvisor} with a higher order (i.e. downstream in the request
		 * direction) is already registered, since that memory advisor will handle history
		 * for every tool-call iteration.
		 */
		private void autoRegisterToolCallingAdvisor() {

			// 即使没有静态注册工具，也保留默认工具 Advisor，以接住其他 Advisor 动态注入的工具。
			// 调用方可通过 advisorParams 显式关闭，或自行提供一个 ToolAdvisor 接管整个循环。
			boolean autoRegisterDisabled = Boolean.FALSE
				.equals(this.advisorParams.get(ChatClientAttributes.TOOL_CALLING_ADVISOR_AUTO_REGISTER.getKey()));
			if (autoRegisterDisabled) {
				return;
			}

			boolean hasToolCallingAdvisor = this.advisors.stream().anyMatch(a -> a instanceof ToolAdvisor);
			if (hasToolCallingAdvisor) {
				return;
			}

			int configuredOrder = this.toolCallingAdvisorBuilder.getAdvisorOrder();

			// 工具 Advisor 每轮只重新执行自己下游的链。若记忆 Advisor 在下游，它每轮都会补历史，
			// 此时关闭工具 Advisor 自己的历史累积，避免两个组件重复装入同一段工具对话。
			boolean hasDownstreamMemoryAdvisor = this.advisors.stream()
				.anyMatch(a -> a instanceof MemoryAdvisor && a.getOrder() > configuredOrder);

			this.advisors.add(this.toolCallingAdvisorBuilder.copy()
				.conversationHistoryEnabled(!hasDownstreamMemoryAdvisor)
				.build());
		}

		private void validateSingleToolAdvisor() {
			List<Advisor> toolAdvisors = this.advisors.stream().filter(a -> a instanceof ToolAdvisor).toList();
			if (toolAdvisors.size() > 1) {
				String names = String.join(", ",
						toolAdvisors.stream().map(a -> a.getName() + " (order=" + a.getOrder() + ")").toList());
				throw new IllegalStateException("At most one ToolAdvisor is allowed in the advisor chain, but found "
						+ toolAdvisors.size() + ": [" + names + "]");
			}
		}

	}

}
