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

package org.springframework.ai.tool.method;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.Type;
import java.util.Map;
import java.util.stream.Stream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jspecify.annotations.Nullable;
import tools.jackson.core.JacksonException;

import org.springframework.ai.chat.model.ToolContext;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.ai.tool.execution.DefaultToolCallResultConverter;
import org.springframework.ai.tool.execution.ToolCallResultConverter;
import org.springframework.ai.tool.execution.ToolExecutionException;
import org.springframework.ai.tool.metadata.ToolMetadata;
import org.springframework.ai.util.JsonHelper;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.util.Assert;
import org.springframework.util.ClassUtils;
import org.springframework.util.CollectionUtils;

/**
 * A {@link ToolCallback} implementation to invoke methods as tools.
 *
 * <p>
 * 中文导读：本类把模型传回的 JSON 参数真正落到一次 Java 反射调用上。
 * 注册时由 MethodToolCallbackProvider 提供工具定义、Method、目标对象和结果转换器；
 * 调用时依次解析 JSON、按 Java 参数名取值、转换具体类型、invoke，再把返回值转换成字符串。
 * 静态方法允许没有目标对象，实例方法必须绑定对象；本类不负责创建业务对象或管理其生命周期。
 *
 * <p>
 * 面向模型的 Schema 已在注册阶段生成，本类执行时不会再次运行 JSON Schema 校验。
 * 例如缺失参数首先表现为 null，之后能否传给 Java 参数由类型转换、反射和业务方法共同决定。
 * ToolContext 则走独立注入分支，不从模型给出的 JSON 字段构造。
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
public final class MethodToolCallback implements ToolCallback {

	private static final JsonHelper jsonHelper = new JsonHelper();

	private static final Log logger = LogFactory.getLog(MethodToolCallback.class);

	private static final ToolCallResultConverter DEFAULT_RESULT_CONVERTER = new DefaultToolCallResultConverter();

	private static final ToolMetadata DEFAULT_TOOL_METADATA = ToolMetadata.builder().build();

	private final ToolDefinition toolDefinition;

	private final ToolMetadata toolMetadata;

	private final Method toolMethod;

	private final @Nullable Object toolObject;

	private final ToolCallResultConverter toolCallResultConverter;

	public MethodToolCallback(ToolDefinition toolDefinition, @Nullable ToolMetadata toolMetadata, Method toolMethod,
			@Nullable Object toolObject, @Nullable ToolCallResultConverter toolCallResultConverter) {
		Assert.notNull(toolDefinition, "toolDefinition cannot be null");
		Assert.notNull(toolMethod, "toolMethod cannot be null");
		Assert.isTrue(Modifier.isStatic(toolMethod.getModifiers()) || toolObject != null,
				"toolObject cannot be null for non-static methods");
		this.toolDefinition = toolDefinition;
		this.toolMetadata = toolMetadata != null ? toolMetadata : DEFAULT_TOOL_METADATA;
		this.toolMethod = toolMethod;
		this.toolObject = toolObject;
		this.toolCallResultConverter = toolCallResultConverter != null ? toolCallResultConverter
				: DEFAULT_RESULT_CONVERTER;
	}

	@Override
	public ToolDefinition getToolDefinition() {
		return this.toolDefinition;
	}

	@Override
	public ToolMetadata getToolMetadata() {
		return this.toolMetadata;
	}

	@Override
	public String call(String toolInput) {
		return call(toolInput, null);
	}

	@Override
	public String call(String toolInput, @Nullable ToolContext toolContext) {
		Assert.hasText(toolInput, "toolInput cannot be null or empty");

		if (logger.isDebugEnabled()) {
			logger.debug("Starting execution of tool: " + this.toolDefinition.name());
		}

		this.validateToolContextSupport(toolContext);

		// JSON 对象的键来自模型按工具 Schema 生成的参数名；先解析为通用 Map，随后恢复各参数类型。
		Map<String, Object> toolArguments = this.extractToolArguments(toolInput);
		Assert.state(toolArguments != null, "toolArguments must not be null");

		Object[] methodArguments = this.buildMethodArguments(toolArguments, toolContext);

		Object result = this.callMethod(methodArguments);

		if (logger.isDebugEnabled()) {
			logger.debug("Successful execution of tool: " + this.toolDefinition.name());
		}

		Type returnType = this.toolMethod.getGenericReturnType();

		// 转换器接收泛型返回类型；默认转换器生成 JSON 字符串，void 则转换为约定的完成响应。
		// 返回值随后成为 ToolResponseMessage 的数据，由工具循环决定回送模型还是直接返回调用方。
		return this.toolCallResultConverter.convert(result, returnType);
	}

	private void validateToolContextSupport(@Nullable ToolContext toolContext) {
		// 当前契约要求声明 ToolContext 的方法收到非空内容，单纯 new ToolContext(emptyMap) 仍不满足。
		// Schema 生成器会排除此特殊参数，因此不能依靠模型补齐它，应用需通过 toolContext 提供。
		var isNonEmptyToolContextProvided = toolContext != null && !CollectionUtils.isEmpty(toolContext.getContext());
		var isToolContextAcceptedByMethod = Stream.of(this.toolMethod.getParameterTypes())
			.anyMatch(type -> ClassUtils.isAssignable(ToolContext.class, type));
		if (isToolContextAcceptedByMethod && !isNonEmptyToolContextProvided) {
			throw new IllegalArgumentException("ToolContext is required by the method as an argument");
		}
	}

	private @Nullable Map<String, Object> extractToolArguments(String toolInput) {
		try {
			return jsonHelper.fromJson(toolInput, new ParameterizedTypeReference<>() {
			});
		}
		catch (Exception ex) {
			logger.warn("Conversion from JSON failed", ex);
			Throwable cause = (ex.getCause() instanceof JacksonException) ? ex.getCause() : ex;
			throw new ToolExecutionException(this.getToolDefinition(), cause);
		}
	}

	// Based on the implementation in MethodToolCallback.
	@SuppressWarnings("null")
	private Object[] buildMethodArguments(Map<String, Object> toolInputArguments, @Nullable ToolContext toolContext) {
		// 按方法声明顺序构造 Object[]，确保 JSON 字段顺序不会影响实际参数位置。
		// parameter.getName() 与 Schema 生成时使用同一个反射参数名，编译时保留参数名尤为重要。
		return Stream.of(this.toolMethod.getParameters()).map(parameter -> {
			if (ClassUtils.isAssignable(ToolContext.class, parameter.getType())) {
				return toolContext;
			}
			Object rawArgument = toolInputArguments.get(parameter.getName());
			return buildTypedArgument(rawArgument, parameter.getParameterizedType());
		}).toArray();
	}

	private @Nullable Object buildTypedArgument(@Nullable Object value, Type type) {
		if (value == null) {
			return null;
		}
		try {
			if (type instanceof Class<?>) {
				// 普通类型直接转换；例如 JSON 数字、字符串或 Map 转成目标参数声明的 Java 类型。
				return jsonHelper.convertToTypedObject(value, (Class<?>) type);
			}

			// For generic types, use the fromJson method that accepts Type
			// 保留 List<Item>、Map<String, Item> 等完整 Type，避免只得到原始 List 或嵌套 Map。
			String json = jsonHelper.toJson(value, true);
			return jsonHelper.fromJson(json, type);
		}
		catch (Exception ex) {
			logger.warn("Conversion from JSON failed", ex);
			Throwable cause = (ex.getCause() instanceof JacksonException) ? ex.getCause() : ex;
			throw new ToolExecutionException(this.getToolDefinition(), cause);
		}
	}

	@SuppressWarnings("NullAway") // ex.getCause() is guaranteed to be non-null
	private @Nullable Object callMethod(Object[] methodArguments) {
		// 对非 public 对象或方法放开反射访问；是否允许仍受 JVM 模块和访问规则约束。
		if (isObjectNotPublic() || isMethodNotPublic()) {
			this.toolMethod.setAccessible(true);
		}

		Object result;
		try {
			// invoke 在当前线程同步执行，没有线程池、事务边界、重试或取消逻辑。
			result = this.toolMethod.invoke(this.toolObject, methodArguments);
		}
		catch (IllegalAccessException ex) {
			throw new IllegalStateException("Could not access method: " + ex.getMessage(), ex);
		}
		catch (InvocationTargetException ex) {
			// 解开反射包装，保留业务异常作为 cause；Manager 的异常处理策略再决定是否回送错误文本。
			// IllegalAccessException 等框架调用失败则走不同分支，不能把所有异常理解成模型可见结果。
			throw new ToolExecutionException(this.toolDefinition, ex.getCause());
		}
		return result;
	}

	private boolean isObjectNotPublic() {
		return this.toolObject != null && !Modifier.isPublic(this.toolObject.getClass().getModifiers());
	}

	private boolean isMethodNotPublic() {
		return !Modifier.isPublic(this.toolMethod.getModifiers());
	}

	@Override
	public String toString() {
		return "MethodToolCallback{" + "toolDefinition=" + this.toolDefinition + ", toolMetadata=" + this.toolMetadata
				+ '}';
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private @Nullable ToolDefinition toolDefinition;

		private @Nullable ToolMetadata toolMetadata;

		private @Nullable Method toolMethod;

		private @Nullable Object toolObject;

		private @Nullable ToolCallResultConverter toolCallResultConverter;

		private Builder() {
		}

		public Builder toolDefinition(ToolDefinition toolDefinition) {
			this.toolDefinition = toolDefinition;
			return this;
		}

		public Builder toolMetadata(ToolMetadata toolMetadata) {
			this.toolMetadata = toolMetadata;
			return this;
		}

		public Builder toolMethod(Method toolMethod) {
			this.toolMethod = toolMethod;
			return this;
		}

		public Builder toolObject(Object toolObject) {
			this.toolObject = toolObject;
			return this;
		}

		public Builder toolCallResultConverter(ToolCallResultConverter toolCallResultConverter) {
			this.toolCallResultConverter = toolCallResultConverter;
			return this;
		}

		@SuppressWarnings("null")
		public MethodToolCallback build() {
			Assert.state(this.toolDefinition != null, "ToolDefinition is required");
			Assert.state(this.toolMethod != null, "ToolMethod is required");
			return new MethodToolCallback(this.toolDefinition, this.toolMetadata, this.toolMethod, this.toolObject,
					this.toolCallResultConverter);
		}

	}

}
