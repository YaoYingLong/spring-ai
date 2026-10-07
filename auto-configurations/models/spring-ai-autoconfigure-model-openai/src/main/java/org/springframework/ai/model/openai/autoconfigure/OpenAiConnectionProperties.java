/*
 * Copyright 2023-2024 the original author or authors.
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

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * 自动解析以下配置
 *  - spring.ai.openai.apiKey
 *  - spring.ai.openai.baseUrl，如果没有配置默认是：https://api.openai.com
 *  - spring.ai.openai.projectId，作为模型调用的header
 *  - spring.ai.openai.organizationId，作为模型调用的header
 */
@ConfigurationProperties(OpenAiConnectionProperties.CONFIG_PREFIX)
public class OpenAiConnectionProperties extends OpenAiParentProperties {

	// spring.ai.openai前缀的配置
	public static final String CONFIG_PREFIX = "spring.ai.openai";
	// 默认的baseURL
	public static final String DEFAULT_BASE_URL = "https://api.openai.com";

	public OpenAiConnectionProperties() {
		super.setBaseUrl(DEFAULT_BASE_URL);
	}

}
