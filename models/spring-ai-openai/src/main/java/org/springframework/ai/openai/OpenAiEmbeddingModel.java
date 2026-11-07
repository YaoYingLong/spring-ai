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
import java.util.List;
import java.util.Objects;

import com.openai.client.OpenAIClient;
import com.openai.core.RequestOptions;
import com.openai.models.embeddings.CreateEmbeddingResponse;
import com.openai.models.embeddings.EmbeddingCreateParams;
import io.micrometer.observation.ObservationRegistry;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jspecify.annotations.Nullable;

import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.MetadataMode;
import org.springframework.ai.embedding.AbstractEmbeddingModel;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.embedding.EmbeddingResponseMetadata;
import org.springframework.ai.embedding.observation.DefaultEmbeddingModelObservationConvention;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationContext;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationConvention;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationDocumentation;
import org.springframework.ai.model.EmbeddingUtils;
import org.springframework.ai.observation.conventions.AiProvider;
import org.springframework.ai.openai.http.okhttp.OpenAiHttpClientBuilderCustomizer;
import org.springframework.ai.openai.setup.OpenAiSetup;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;

/**
 * Embedding Model implementation using the OpenAI Java SDK.
 * <p>
 * 中文导读：把 Spring AI 的文本批次转换成 OpenAI embeddings 请求，再把返回的数值列表
 * 转成 {@code float[]}。模型只负责向量计算，不负责切分文档、保存向量或执行相似度检索。
 * <p>
 * 主链是 {@code embed(Document) -> call(EmbeddingRequest) -> embeddings().create}
 * 再到 {@code generateEmbeddingResponse}；批量调用直接传入包含多条文本的 EmbeddingRequest。
 * 文档走 {@link MetadataMode} 格式化后才变成模型输入，元数据是否参与向量内容由该模式决定。
 * <p>
 * 请求 options 通过 builder 合并到模型默认 options，具体字段遵守
 * {@link OpenAiEmbeddingOptions.Builder#merge} 的规则；连接客户端只在构造时确定。
 * {@code dimensions()} 继承父类的探测与缓存逻辑，首次查询可能触发一次真实 embedding 调用；
 * 请求中的 dimensions 参数只传给服务端，本类不在本地截断或补齐返回向量。
 *
 * @author Julien Dubois
 * @author Soby Chacko
 * @author Thomas Vitale
 * @author Christian Tzolov
 * @author Josh Long
 * @author guan xu
 */
public class OpenAiEmbeddingModel extends AbstractEmbeddingModel {

	private static final EmbeddingModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultEmbeddingModelObservationConvention();

	private static final Log logger = LogFactory.getLog(OpenAiEmbeddingModel.class);

	private final OpenAIClient openAiClient;

	private final OpenAiEmbeddingOptions options;

	private final MetadataMode metadataMode;

	private final ObservationRegistry observationRegistry;

	private EmbeddingModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	/**
	 * Creates a new OpenAiEmbeddingModel with default options.
	 * @deprecated in favor of {@link OpenAiEmbeddingModel#builder()}
	 */
	@Deprecated
	public OpenAiEmbeddingModel() {
		this(null, null, null, null);
	}

	/**
	 * Creates a new OpenAiEmbeddingModel with the given options.
	 * @param options the embedding options
	 * @deprecated in favor of {@link OpenAiEmbeddingModel#builder()}
	 */
	@Deprecated
	public OpenAiEmbeddingModel(@Nullable OpenAiEmbeddingOptions options) {
		this(null, null, options, null);
	}

	/**
	 * Creates a new OpenAiEmbeddingModel with the given metadata mode and options.
	 * @param metadataMode the metadata mode
	 * @param options the embedding options
	 * @deprecated in favor of {@link OpenAiEmbeddingModel#builder()}
	 */
	@Deprecated
	public OpenAiEmbeddingModel(@Nullable MetadataMode metadataMode, @Nullable OpenAiEmbeddingOptions options) {
		this(null, metadataMode, options, null);
	}

	/**
	 * Creates a new OpenAiEmbeddingModel with the given options and observation registry.
	 * @param options the embedding options
	 * @param observationRegistry the observation registry
	 * @deprecated in favor of {@link OpenAiEmbeddingModel#builder()}
	 */
	@Deprecated
	public OpenAiEmbeddingModel(@Nullable OpenAiEmbeddingOptions options,
			@Nullable ObservationRegistry observationRegistry) {
		this(null, null, options, observationRegistry);
	}

	/**
	 * Creates a new OpenAiEmbeddingModel with the given metadata mode, options, and
	 * observation registry.
	 * @param metadataMode the metadata mode
	 * @param options the embedding options
	 * @param observationRegistry the observation registry
	 * @deprecated in favor of {@link OpenAiEmbeddingModel#builder()}
	 */
	@Deprecated
	public OpenAiEmbeddingModel(@Nullable MetadataMode metadataMode, @Nullable OpenAiEmbeddingOptions options,
			@Nullable ObservationRegistry observationRegistry) {
		this(null, metadataMode, options, observationRegistry);
	}

	/**
	 * Creates a new OpenAiEmbeddingModel with the given OpenAI client.
	 * @param openAiClient the OpenAI client
	 * @deprecated in favor of {@link OpenAiEmbeddingModel#builder()}
	 */
	@Deprecated
	public OpenAiEmbeddingModel(@Nullable OpenAIClient openAiClient) {
		this(openAiClient, null, null, null);
	}

	/**
	 * Creates a new OpenAiEmbeddingModel with the given OpenAI client and metadata mode.
	 * @param openAiClient the OpenAI client
	 * @param metadataMode the metadata mode
	 * @deprecated in favor of {@link OpenAiEmbeddingModel#builder()}
	 */
	@Deprecated
	public OpenAiEmbeddingModel(@Nullable OpenAIClient openAiClient, @Nullable MetadataMode metadataMode) {
		this(openAiClient, metadataMode, null, null);
	}

	/**
	 * Creates a new OpenAiEmbeddingModel with all configuration options.
	 * @param openAiClient the OpenAI client
	 * @param metadataMode the metadata mode
	 * @param options the embedding options
	 * @deprecated in favor of {@link OpenAiEmbeddingModel#builder()}
	 */
	@Deprecated
	public OpenAiEmbeddingModel(@Nullable OpenAIClient openAiClient, @Nullable MetadataMode metadataMode,
			@Nullable OpenAiEmbeddingOptions options) {
		this(openAiClient, metadataMode, options, null);
	}

	/**
	 * Creates a new OpenAiEmbeddingModel with all configuration options.
	 * @param openAiClient the OpenAI client
	 * @param metadataMode the metadata mode
	 * @param options the embedding options
	 * @param observationRegistry the observation registry
	 * @deprecated in favor of {@link OpenAiEmbeddingModel#builder()}
	 */
	@Deprecated
	public OpenAiEmbeddingModel(@Nullable OpenAIClient openAiClient, @Nullable MetadataMode metadataMode,
			@Nullable OpenAiEmbeddingOptions options, @Nullable ObservationRegistry observationRegistry) {

		this(builder().openAiClient(openAiClient)
			.metadataMode(metadataMode)
			.options(options)
			.observationRegistry(observationRegistry));
	}

	public static Builder builder() {
		return new Builder();
	}

	private OpenAiEmbeddingModel(Builder builder) {
		// 默认把文档用于 EMBED 的元数据纳入格式化内容；没有注册表时仍可通过 NOOP 保持同一调用链。
		this.options = Objects.requireNonNullElseGet(builder.options, () -> OpenAiEmbeddingOptions.builder().build());
		this.metadataMode = Objects.requireNonNullElse(builder.metadataMode, MetadataMode.EMBED);
		this.observationRegistry = Objects.requireNonNullElse(builder.observationRegistry, ObservationRegistry.NOOP);
		// 用户提供客户端则直接复用；否则在这里一次性确定地址、凭据、超时、重试与 HTTP 定制。
		this.openAiClient = Objects.requireNonNullElseGet(builder.openAiClient,
				() -> OpenAiSetup.setupSyncClient(this.options.getBaseUrl(), this.options.getApiKey(),
						this.options.getCredential(), this.options.getMicrosoftDeploymentName(),
						this.options.getMicrosoftFoundryServiceVersion(), this.options.getOrganizationId(),
						this.options.isMicrosoftFoundry(), this.options.isGitHubModels(), this.options.getModel(),
						Objects.requireNonNullElse(this.options.getTimeout(), AbstractOpenAiOptions.DEFAULT_TIMEOUT),
						this.options.getMaxRetries(), this.options.getProxy(), this.options.getCustomHeaders(),
						this.observationRegistry, null, builder.httpClientCustomizers));
	}

	@Override
	public String getEmbeddingContent(Document document) {
		Assert.notNull(document, "Document must not be null");
		// 批量文档入口也可使用此方法，保证文本格式化与下面单文档 embed 的策略一致。
		return document.getFormattedContent(this.metadataMode);
	}

	@Override
	public float[] embed(Document document) {
		// 单文档只是批量协议的一条输入；响应为空时返回空数组，否则取首条向量。
		EmbeddingResponse response = this
			.call(new EmbeddingRequest(List.of(document.getFormattedContent(this.metadataMode)), this.options));

		if (CollectionUtils.isEmpty(response.getResults())) {
			return new float[0];
		}
		return response.getResults().get(0).getOutput();
	}

	@Override
	public EmbeddingResponse call(EmbeddingRequest embeddingRequest) {
		// 1. 复制模型选项，再按 merge 规则合入本次选项；model/dimensions 等非空值可覆盖默认值。
		// OpenAI 专有字段另有规则，例如 headers/extraBody 按键合并，timeout 直接采用传入对象的值。
		OpenAiEmbeddingOptions options = OpenAiEmbeddingOptions.builder()
			.from(this.options)
			.merge(embeddingRequest.getOptions())
			.build();

		EmbeddingRequest embeddingRequestWithMergedOptions = new EmbeddingRequest(embeddingRequest.getInstructions(),
				options);

		// 2. 将 List<String> 作为一个 SDK 批次输入，并映射模型/部署名、编码格式、维度和扩展参数。
		// 此方法不按 token 数拆批；调用方或上层 VectorStore 的批处理策略决定每次请求包含哪些文本。
		EmbeddingCreateParams embeddingCreateParams = options
			.toOpenAiCreateParams(embeddingRequestWithMergedOptions.getInstructions());

		if (logger.isTraceEnabled()) {
			logger.trace("OpenAiEmbeddingModel call " + options.getModel() + " with the following options : "
					+ embeddingCreateParams);
		}

		// 3. 仅 timeout 进入 SDK 请求级 RequestOptions；合并 options 不会重新建立连接客户端。
		RequestOptions requestOptions = this.buildRequestOptions(options);

		var observationContext = EmbeddingModelObservationContext.builder()
			.embeddingRequest(embeddingRequestWithMergedOptions)
			.provider(AiProvider.OPENAI.value())
			.build();

		// 4. 同步模型 Observation 包住 SDK 调用和结果转换，记录模型、输入与响应用量。
		// 调用/转换异常沿当前线程向外传播；本层没有额外重试循环，SDK 使用客户端构造时的重试设置。
		return Objects.requireNonNull(
				EmbeddingModelObservationDocumentation.EMBEDDING_MODEL_OPERATION
					.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
							this.observationRegistry)
					.observe(() -> {
						CreateEmbeddingResponse response = this.openAiClient.embeddings()
							.create(embeddingCreateParams, requestOptions);

						var embeddingResponse = generateEmbeddingResponse(response);
						observationContext.setResponse(embeddingResponse);
						return embeddingResponse;
					}));
	}

	/**
	 * Creates a RequestOptions instance from the given embedding options.
	 * <p>
	 * 未提供请求超时时让 SDK 沿用客户端默认值；API key/base URL 等连接配置不在这里生效。
	 * @param options the embedding options
	 * @return a RequestOptions instance
	 */
	private RequestOptions buildRequestOptions(OpenAiEmbeddingOptions options) {
		Assert.notNull(options, "Options cannot be null");
		RequestOptions.Builder requestOptionsBuilder = RequestOptions.builder();
		if (options.getTimeout() != null) {
			requestOptionsBuilder.timeout(options.getTimeout());
		}
		return requestOptionsBuilder.build();
	}

	private EmbeddingResponse generateEmbeddingResponse(CreateEmbeddingResponse response) {

		// 每条结果保留服务端 index；批次级 model/usage 单独保存在响应元数据中。
		List<Embedding> data = generateEmbeddingList(response.data());
		EmbeddingResponseMetadata metadata = new EmbeddingResponseMetadata();
		metadata.setModel(response.model());
		metadata.setUsage(getDefaultUsage(response.usage()));
		return new EmbeddingResponse(data, metadata);
	}

	private DefaultUsage getDefaultUsage(CreateEmbeddingResponse.Usage nativeUsage) {
		// Embedding 没有文本生成用量，因此 completionTokens 为 0；输入/总量取服务端统计。
		return new DefaultUsage(toIntTokenCount("promptTokens", nativeUsage.promptTokens()), 0,
				toIntTokenCount("totalTokens", nativeUsage.totalTokens()), nativeUsage);
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

	private List<Embedding> generateEmbeddingList(List<com.openai.models.embeddings.Embedding> nativeData) {
		// SDK 已把响应解析为 List<Float>，这里拆箱成框架的 float[]，数组长度就是实际返回维度。
		// 保持服务端结果顺序，不在此按 index 重排；index 用精确转换，超出 int 范围时失败。
		List<Embedding> data = new ArrayList<>();
		for (com.openai.models.embeddings.Embedding nativeDatum : nativeData) {
			List<Float> nativeDatumEmbedding = nativeDatum.embedding();
			long nativeIndex = nativeDatum.index();
			Embedding embedding = new Embedding(EmbeddingUtils.toPrimitive(nativeDatumEmbedding),
					Math.toIntExact(nativeIndex));
			data.add(embedding);
		}
		return data;
	}

	/**
	 * Gets the embedding options for this model.
	 * @return the embedding options
	 */
	public OpenAiEmbeddingOptions getOptions() {
		return this.options;
	}

	/**
	 * Use the provided convention for reporting observation data
	 * @param observationConvention The provided convention
	 */
	public void setObservationConvention(EmbeddingModelObservationConvention observationConvention) {
		Assert.notNull(observationConvention, "observationConvention cannot be null");
		this.observationConvention = observationConvention;
	}

	/**
	 * 装配默认选项、文档格式化策略和可复用客户端；build 只创建适配器，推理在 call/embed 时发生。
	 */
	public static final class Builder {

		private @Nullable OpenAIClient openAiClient;

		private @Nullable OpenAiEmbeddingOptions options;

		private @Nullable MetadataMode metadataMode;

		private @Nullable ObservationRegistry observationRegistry;

		private List<OpenAiHttpClientBuilderCustomizer> httpClientCustomizers = new ArrayList<>();

		private Builder() {
		}

		public Builder openAiClient(@Nullable OpenAIClient openAiClient) {
			this.openAiClient = openAiClient;
			return this;
		}

		public Builder options(@Nullable OpenAiEmbeddingOptions options) {
			this.options = options;
			return this;
		}

		public Builder metadataMode(@Nullable MetadataMode metadataMode) {
			this.metadataMode = metadataMode;
			return this;
		}

		public Builder observationRegistry(@Nullable ObservationRegistry observationRegistry) {
			this.observationRegistry = observationRegistry;
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

		public OpenAiEmbeddingModel build() {
			return new OpenAiEmbeddingModel(this);
		}

	}

}
