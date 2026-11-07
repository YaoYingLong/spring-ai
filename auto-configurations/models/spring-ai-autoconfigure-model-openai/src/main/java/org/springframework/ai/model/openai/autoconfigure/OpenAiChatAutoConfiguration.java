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

package org.springframework.ai.model.openai.autoconfigure;

import com.openai.client.OpenAIClient;
import com.openai.client.OpenAIClientAsync;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.observation.ObservationRegistry;
import org.jspecify.annotations.Nullable;
import org.springframework.ai.chat.observation.ChatModelObservationConvention;
import org.springframework.ai.model.SpringAIModelProperties;
import org.springframework.ai.model.SpringAIModels;
import org.springframework.ai.model.openai.autoconfigure.OpenAiAutoConfigurationUtil.ResolvedConnectionProperties;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.http.okhttp.OpenAiHttpClientBuilderCustomizer;
import org.springframework.ai.openai.setup.OpenAiSetup;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

import java.util.List;

/**
 * Chat {@link AutoConfiguration Auto-configuration} for OpenAI SDK.
 * <p>
 * 中文导读：把外部配置、SDK 连接客户端、工具定义解析器和观测设施装配成 OpenAiChatModel。
 * 类级条件检查 {@code spring.ai.model.chat=openai}，未配置时也匹配；
 * Bean 级条件再检查 {@code spring.ai.openai.chat.api=chat-completions}，同样是默认分支。
 * <p>
 * 配置 {@code spring.ai.openai.chat.api=responses} 时，本 Bean 方法不生效，
 * 由 {@link OpenAiResponsesChatAutoConfiguration} 创建另一种 ChatModel 实现。
 * API 分支选择与 provider 选择是两层条件，不能只更改模型名称来切换端点。
 * <p>
 * 无参数的 ConditionalOnMissingBean 按本方法的返回类型 OpenAiChatModel 推断，
 * 并非检查任意 ChatModel 都存在就退出。装配 ChatClient 时仍需容器能确定要注入的模型。
 *
 * @author Christian Tzolov
 * @author Soby Chacko
 * @author Thomas Vitale
 * @author Stefan Vassilev
 * @author Yanming Zhou
 * @author Issam El-atif
 * @author Ilayaperumal Gopinathan
 * @author Sebastien Deleuze
 */
@AutoConfiguration
@EnableConfigurationProperties({ OpenAiCommonProperties.class, OpenAiChatProperties.class })
@ConditionalOnProperty(name = SpringAIModelProperties.CHAT_MODEL, havingValue = SpringAIModels.OPENAI,
		matchIfMissing = true)
public class OpenAiChatAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean
	@ConditionalOnProperty(name = OpenAiChatProperties.CONFIG_PREFIX + ".api", havingValue = "chat-completions",
			matchIfMissing = true)
	public OpenAiChatModel openAiChatModel(
			OpenAiCommonProperties commonProperties,
			OpenAiChatProperties chatProperties,
			ToolCallingManager toolCallingManager,
			ObjectProvider<ObservationRegistry> observationRegistry,
			ObjectProvider<MeterRegistry> meterRegistry,
			ObjectProvider<ChatModelObservationConvention> observationConvention,
			ObjectProvider<OpenAiHttpClientBuilderCustomizer> httpClientBuilderCustomizers) {

		// 1. 汇总通用连接属性与 chat 专用覆盖值；实际优先级集中在 resolveCommonProperties。
		// 例如 apiKey 为 null 才回退，显式空字符串保留无认证语义；布尔标志和重试值有各自合并规则。
		var resolvedProperties = OpenAiAutoConfigurationUtil.resolveCommonProperties(commonProperties, chatProperties);

		// 连接池指标与模型 Observation 分别装配；没有 MeterRegistry 时仍可创建模型。
		MeterRegistry meterRegistryToUse = resolvedProperties.isConnectionPoolMetricsEnabled()
				? meterRegistry.getIfAvailable() : null;

		// 保留 Spring 的排序顺序；OpenAiSetup 在设置 HTTP 默认项之后依次执行这些定制器。
		List<OpenAiHttpClientBuilderCustomizer> customizers = httpClientBuilderCustomizers.orderedStream().toList();

		// 2. 同步与异步客户端分别创建并由模型长期复用；这些私有工厂方法本身不注册额外 Bean。
		OpenAIClient openAIClient = this.openAiClient(resolvedProperties, observationRegistry, meterRegistryToUse,
				customizers);

		OpenAIClientAsync openAIClientAsync = this.openAiClientAsync(resolvedProperties, observationRegistry,
				meterRegistryToUse, customizers);

		// 3. 默认生成参数来自 chatProperties.toOptions，连接参数已交给上面的 SDK 客户端。
		// ToolCallingManager 在模型内只解析工具定义；真正的工具执行循环由 ChatClient 的 Advisor 负责。
		// 观测注册表没有唯一候选时使用 NOOP，保持不依赖可观测设施的基本调用能力。
		var chatModel = OpenAiChatModel.builder()
			.openAiClient(openAIClient)
			.openAiClientAsync(openAIClientAsync)
			.options(chatProperties.toOptions())
			.toolCallingManager(toolCallingManager)
			.observationRegistry(observationRegistry.getIfUnique(() -> ObservationRegistry.NOOP))
			.meterRegistry(meterRegistryToUse)
			.build();

		// 有可注入的自定义 Convention 时覆盖模型默认观测命名/标签约定。
		observationConvention.ifAvailable(chatModel::setObservationConvention);

		return chatModel;
	}

	private OpenAIClient openAiClient(ResolvedConnectionProperties commonProperties,
			ObjectProvider<ObservationRegistry> observationRegistry, @Nullable MeterRegistry meterRegistry,
			List<OpenAiHttpClientBuilderCustomizer> httpClientCustomizers) {

		// 重试次数在 SDK 客户端构建阶段配置，模型的一次 call 不再额外包一层 Spring 重试循环。
		return OpenAiSetup.setupSyncClient(commonProperties.getBaseUrl(), commonProperties.getApiKey(),
				commonProperties.getCredential(), commonProperties.getMicrosoftDeploymentName(),
				commonProperties.getMicrosoftFoundryServiceVersion(), commonProperties.getOrganizationId(),
				commonProperties.isMicrosoftFoundry(), commonProperties.isGitHubModels(), commonProperties.getModel(),
				commonProperties.getTimeout(), commonProperties.getMaxRetries(), commonProperties.getProxy(),
				commonProperties.getCustomHeaders(), observationRegistry.getIfUnique(() -> ObservationRegistry.NOOP),
				meterRegistry, httpClientCustomizers);
	}

	private OpenAIClientAsync openAiClientAsync(ResolvedConnectionProperties commonProperties,
			ObjectProvider<ObservationRegistry> observationRegistry, @Nullable MeterRegistry meterRegistry,
			List<OpenAiHttpClientBuilderCustomizer> httpClientCustomizers) {

		// 流式接口由该异步客户端提供回调；每次订阅只关闭自己的流响应，不在此按请求创建客户端。
		return OpenAiSetup.setupAsyncClient(commonProperties.getBaseUrl(), commonProperties.getApiKey(),
				commonProperties.getCredential(), commonProperties.getMicrosoftDeploymentName(),
				commonProperties.getMicrosoftFoundryServiceVersion(), commonProperties.getOrganizationId(),
				commonProperties.isMicrosoftFoundry(), commonProperties.isGitHubModels(), commonProperties.getModel(),
				commonProperties.getTimeout(), commonProperties.getMaxRetries(), commonProperties.getProxy(),
				commonProperties.getCustomHeaders(), observationRegistry.getIfUnique(() -> ObservationRegistry.NOOP),
				meterRegistry, httpClientCustomizers);
	}

}
