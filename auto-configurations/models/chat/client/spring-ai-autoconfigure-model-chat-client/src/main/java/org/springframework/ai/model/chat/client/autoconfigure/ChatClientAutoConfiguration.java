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

package org.springframework.ai.model.chat.client.autoconfigure;

import io.micrometer.observation.ObservationRegistry;
import io.micrometer.tracing.Tracer;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.springframework.ai.chat.client.AdvisorParams;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClientBuilderCustomizer;
import org.springframework.ai.chat.client.ChatClientCustomizer;
import org.springframework.ai.chat.client.advisor.ToolCallingAdvisor;
import org.springframework.ai.chat.client.advisor.observation.AdvisorObservationConvention;
import org.springframework.ai.chat.client.observation.ChatClientCompletionObservationHandler;
import org.springframework.ai.chat.client.observation.ChatClientObservationContext;
import org.springframework.ai.chat.client.observation.ChatClientObservationConvention;
import org.springframework.ai.chat.client.observation.ChatClientPromptContentObservationHandler;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.ToolExecutionEligibilityChecker;
import org.springframework.ai.model.tool.autoconfigure.ToolCallingAutoConfiguration;
import org.springframework.ai.observation.TracingAwareLoggingObservationHandler;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

/**
 * {@link EnableAutoConfiguration Auto-configuration} for {@link ChatClient}.
 * <p>
 * This will produce a {@link ChatClient.Builder ChatClient.Builder} bean with the
 * {@code prototype} scope, meaning each injection point will receive a newly cloned
 * instance of the builder.
 * <p>
 * 中文导读：这一层把底层 ChatModel、客户端/Advisor 观测和定制器接到 ChatClient.Builder。
 * 默认启用，可用 {@code spring.ai.chat.client.enabled=false} 关闭；
 * 各 Bean 的 ConditionalOnMissingBean 允许应用替换对应装配部件。
 * <p>
 * 自动配置提供的是 prototype Builder，应用通常注入后再调用 build() 创建 ChatClient。
 * 每次从容器获取 Builder 都运行本工厂方法，避免不同注入点共享可变的默认提示词/Advisor 设置。
 * 创建 Builder 时必须能解析出要注入的 ChatModel，本类不在多个 provider 之间自行挑选模型。
 * <p>
 * ToolCallingAutoConfiguration 先准备工具管理器，本类再把它接入 ToolCallingAdvisor 的构建器；
 * 因而本地工具执行与多轮模型调用属于客户端 Advisor 链，而非供应商模型适配器。
 *
 * @author Christian Tzolov
 * @author Mark Pollack
 * @author Josh Long
 * @author Arjen Poutsma
 * @author Thomas Vitale
 * @author Jonatan Ivanov
 * @since 1.0.0
 */
@AutoConfiguration(after = ToolCallingAutoConfiguration.class)
@ConditionalOnClass(ChatClient.class)
@EnableConfigurationProperties(ChatClientBuilderProperties.class)
@ConditionalOnProperty(prefix = ChatClientBuilderProperties.CONFIG_PREFIX, name = "enabled", havingValue = "true",
		matchIfMissing = true)
public class ChatClientAutoConfiguration {

	private static final Log logger = LogFactory.getLog(ChatClientAutoConfiguration.class);

	private static void logPromptContentWarning() {
		logger.warn(
				"You have enabled logging out the ChatClient prompt content with the risk of exposing sensitive or private information. Please, be careful!");
	}

	private static void logCompletionWarning() {
		logger.warn(
				"You have enabled logging out the ChatClient completion content with the risk of exposing sensitive or private information. Please, be careful!");
	}

	@Bean
	@ConditionalOnMissingBean
	@SuppressWarnings("removal")
	ChatClientBuilderConfigurer chatClientBuilderConfigurer(ObjectProvider<ChatClientCustomizer> customizerProvider,
			ObjectProvider<ChatClientBuilderCustomizer> builderCustomizerProvider) {
		// 各类定制器分别按 Spring 顺序收集；Configurer 先应用旧 customizer，再应用新的 builder customizer。
		ChatClientBuilderConfigurer configurer = new ChatClientBuilderConfigurer();
		configurer.setChatClientCustomizers(customizerProvider.orderedStream().toList());
		configurer.setChatClientBuilderCustomizers(builderCustomizerProvider.orderedStream().toList());
		return configurer;
	}

	@Bean
	@ConditionalOnMissingBean
	@ConditionalOnBean(ToolCallingManager.class)
	ToolCallingAdvisor.Builder<?> toolCallingAdvisorBuilder(ChatClientBuilderProperties properties,
			ToolCallingManager toolCallingManager,
			ObjectProvider<ToolExecutionEligibilityChecker> toolExecutionEligibilityChecker) {
		// 复用容器中的管理器，保留其工具解析、异常策略、调用次数限制和观测配置。
		// eligibility checker 决定一轮响应是否需要执行工具；order 决定此 Advisor 在链中的位置。
		var builder = ToolCallingAdvisor.builder()
			.toolCallingManager(toolCallingManager)
			.advisorOrder(properties.getToolCalling().getAdvisorOrder());

		toolExecutionEligibilityChecker.ifAvailable(builder::toolExecutionEligibilityChecker);

		return builder;
	}

	@Bean
	@Scope("prototype")
	@ConditionalOnMissingBean
	ChatClient.Builder chatClientBuilder(ChatClientBuilderProperties properties,
			ChatClientBuilderConfigurer chatClientBuilderConfigurer, ChatModel chatModel,
			ObjectProvider<ObservationRegistry> observationRegistry,
			ObjectProvider<ChatClientObservationConvention> chatClientObservationConvention,
			ObjectProvider<AdvisorObservationConvention> advisorObservationConvention,
			ObjectProvider<ToolCallingAdvisor.Builder<?>> toolCallingAdvisorBuilder) {
		// 1. 为这次容器获取创建新 Builder，同时传入模型、两层观测 Convention 和工具 Advisor 构建器。
		// 这些 Observation 分别描述客户端请求、Advisor 和模型调用，不应理解为同一个记录对象。
		ChatClient.Builder builder = ChatClient.builder(chatModel,
				observationRegistry.getIfUnique(() -> ObservationRegistry.NOOP),
				chatClientObservationConvention.getIfUnique(), advisorObservationConvention.getIfUnique(),
				toolCallingAdvisorBuilder.getIfAvailable());
		if (!properties.getToolCalling().isEnabled()) {
			// 开关控制默认工具 Advisor 的自动注册，不删除容器中的工具定义或 ToolCallingManager。
			builder.defaultAdvisors(AdvisorParams.toolCallingAdvisorAutoRegister(false));
		}
		// 2. 最后应用用户定制器；应用拿到 Builder 后还可继续设置并决定何时 build。
		return chatClientBuilderConfigurer.configure(builder);
	}

	// Tracer 类与 Bean 同时存在才走此分支，内容日志处理器通过包装器关联当前 trace。
	@Configuration(proxyBeanMethods = false)
	@ConditionalOnClass(Tracer.class)
	@ConditionalOnBean(Tracer.class)
	static class TracerPresentObservationConfiguration {

		@Bean
		@ConditionalOnMissingBean(value = ChatClientPromptContentObservationHandler.class,
				name = "chatClientPromptContentObservationHandler")
		@ConditionalOnProperty(prefix = ChatClientBuilderProperties.CONFIG_PREFIX + ".observations",
				name = "log-prompt", havingValue = "true")
		TracingAwareLoggingObservationHandler<ChatClientObservationContext> chatClientPromptContentObservationHandler(
				Tracer tracer) {
			logPromptContentWarning();
			return new TracingAwareLoggingObservationHandler<>(new ChatClientPromptContentObservationHandler(), tracer);
		}

		@Bean
		@ConditionalOnMissingBean(value = ChatClientCompletionObservationHandler.class,
				name = "chatClientCompletionObservationHandler")
		@ConditionalOnProperty(prefix = ChatClientBuilderProperties.CONFIG_PREFIX + ".observations",
				name = "log-completion", havingValue = "true")
		TracingAwareLoggingObservationHandler<ChatClientObservationContext> chatClientCompletionObservationHandler(
				Tracer tracer) {
			logCompletionWarning();
			return new TracingAwareLoggingObservationHandler<>(new ChatClientCompletionObservationHandler(), tracer);
		}

	}

	// 此备用分支要求 Tracer 类本身缺失；并非“有 Tracer 类但无 Bean”时也自动走这里。
	// 两种日志都需要显式开启相应属性，默认装配 Builder 不会启用提示词/回答内容日志。
	@Configuration(proxyBeanMethods = false)
	@ConditionalOnMissingClass("io.micrometer.tracing.Tracer")
	static class TracerNotPresentObservationConfiguration {

		@Bean
		@ConditionalOnMissingBean
		@ConditionalOnProperty(prefix = ChatClientBuilderProperties.CONFIG_PREFIX + ".observations",
				name = "log-prompt", havingValue = "true")
		ChatClientPromptContentObservationHandler chatClientPromptContentObservationHandler() {
			logPromptContentWarning();
			return new ChatClientPromptContentObservationHandler();
		}

		@Bean
		@ConditionalOnMissingBean
		@ConditionalOnProperty(prefix = ChatClientBuilderProperties.CONFIG_PREFIX + ".observations",
				name = "log-completion", havingValue = "true")
		ChatClientCompletionObservationHandler chatClientCompletionObservationHandler() {
			logCompletionWarning();
			return new ChatClientCompletionObservationHandler();
		}

	}

}
