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

package org.springframework.ai.chat.client.advisor.api;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.AdvisorUtils;
import org.springframework.util.Assert;

/**
 * Base advisor that implements common aspects of the {@link CallAdvisor} and
 * {@link StreamAdvisor}, reducing the boilerplate code needed to implement an advisor.
 * <p>
 * It provides default implementations for the
 * {@link #adviseCall(ChatClientRequest, CallAdvisorChain)} and
 * {@link #adviseStream(ChatClientRequest, StreamAdvisorChain)} methods, delegating the
 * actual logic to the {@link #before(ChatClientRequest, AdvisorChain advisorChain)} and
 * {@link #after(ChatClientResponse, AdvisorChain advisorChain)} methods.
 * <p>
 * 中文导读：将常见 Advisor 写成 before/after 模板，便于在调用前改写 Prompt，
 * 在调用后补充响应。同步路径是 before、下游链、after 的顺序，异常会中断正常返回路径。
 * <p>
 * 流式默认实现先在 getScheduler() 指定的线程池执行 before，再订阅下游。
 * after 仅作用于包含结束原因的响应片段，不自动获得整段回答，也不是 doFinally。
 * 需要完整响应或特殊取消处理的 Advisor 应自行实现流式聚合/生命周期逻辑。
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
public interface BaseAdvisor extends CallAdvisor, StreamAdvisor {

	Scheduler DEFAULT_SCHEDULER = Schedulers.boundedElastic();

	@Override
	default ChatClientResponse adviseCall(ChatClientRequest chatClientRequest, CallAdvisorChain callAdvisorChain) {
		Assert.notNull(chatClientRequest, "chatClientRequest cannot be null");
		Assert.notNull(callAdvisorChain, "callAdvisorChain cannot be null");

		ChatClientRequest processedChatClientRequest = before(chatClientRequest, callAdvisorChain);
		ChatClientResponse chatClientResponse = callAdvisorChain.nextCall(processedChatClientRequest);
		return after(chatClientResponse, callAdvisorChain);
	}

	@Override
	default Flux<ChatClientResponse> adviseStream(ChatClientRequest chatClientRequest,
			StreamAdvisorChain streamAdvisorChain) {
		Assert.notNull(chatClientRequest, "chatClientRequest cannot be null");
		Assert.notNull(streamAdvisorChain, "streamAdvisorChain cannot be null");
		Assert.notNull(getScheduler(), "scheduler cannot be null");

		Flux<ChatClientResponse> chatClientResponseFlux = Mono.just(chatClientRequest)
			.publishOn(getScheduler())
			.map(request -> this.before(request, streamAdvisorChain))
			.flatMapMany(streamAdvisorChain::nextStream);

		return chatClientResponseFlux.map(response -> {
			// 这里判断响应的 finishReason，而非“每个 token 都执行 after”或“流结束后执行一次”。
			if (AdvisorUtils.onFinishReason().test(response)) {
				response = after(response, streamAdvisorChain);
			}
			return response;
		}).onErrorResume(error -> Flux.error(new IllegalStateException("Stream processing failed", error)));
	}

	@Override
	default String getName() {
		return this.getClass().getSimpleName();
	}

	/**
	 * Logic to be executed before the rest of the advisor chain is called.
	 */
	ChatClientRequest before(ChatClientRequest chatClientRequest, AdvisorChain advisorChain);

	/**
	 * Logic to be executed after the rest of the advisor chain is called.
	 */
	ChatClientResponse after(ChatClientResponse chatClientResponse, AdvisorChain advisorChain);

	/**
	 * Scheduler used for processing the advisor logic when streaming.
	 */
	default Scheduler getScheduler() {
		return DEFAULT_SCHEDULER;
	}

}
