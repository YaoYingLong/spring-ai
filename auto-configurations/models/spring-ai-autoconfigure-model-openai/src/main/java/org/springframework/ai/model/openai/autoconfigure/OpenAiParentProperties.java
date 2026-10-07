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

/**
 * Internal parent properties for the OpenAI properties.
 *
 * @author Christian Tzolov
 * @since 0.8.0
 *
 * 这是一个公共的基础类，被各种带有@ConfigurationProperties注解的具体类继承，添加不同的前缀用于解析不同前缀的apiKey、baseUrl等字段
 * 例如：OpenAiConnectionProperties会自动解析
 * - spring.ai.openai.apiKey
 * - spring.ai.openai.baseUrl
 * - spring.ai.openai.projectId
 * - spring.ai.openai.organizationId
 */
class OpenAiParentProperties {

	private String apiKey;

	private String baseUrl;

	private String projectId;

	private String organizationId;

	public String getApiKey() {
		return this.apiKey;
	}

	public void setApiKey(String apiKey) {
		this.apiKey = apiKey;
	}

	public String getBaseUrl() {
		return this.baseUrl;
	}

	public void setBaseUrl(String baseUrl) {
		this.baseUrl = baseUrl;
	}

	public String getProjectId() {
		return this.projectId;
	}

	public void setProjectId(String projectId) {
		this.projectId = projectId;
	}

	public String getOrganizationId() {
		return this.organizationId;
	}

	public void setOrganizationId(String organizationId) {
		this.organizationId = organizationId;
	}

}
