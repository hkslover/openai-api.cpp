#include "openai_api/encoder/encoder.hpp"

#include <iostream>
#include <cassert>
#include <map>
#include "utils/json.hpp"

using namespace openai_api;

// ============================================================
// Helper: strip SSE envelope ("data: {json}\n\n" → json string)
// ============================================================
static std::string sse_to_json_str(const std::string& sse) {
    // Format: "data: <json>\n\n"
    // Remove "data: " prefix (6 chars) and trailing "\n\n" (2 chars)
    assert(sse.find("data: ") == 0);
    assert(sse.size() >= 8);
    return sse.substr(6, sse.size() - 8);
}

// ============================================================
// Test 1: SSE encoding of single ToolCallDelta
// ============================================================
void test_sse_single_tool_call_delta() {
    std::cout << "Test: sse_single_tool_call_delta... " << std::flush;

    ChatCompletionsSSEEncoder encoder;

    auto chunk = OutputChunk::ToolCallDelta(
        "call_abcd1234", "get_weather", R"({"location": "Beijing"})", 0);
    chunk.id = "test-id-1";
    chunk.created = 1234567890;

    std::string encoded = encoder.encode(chunk);

    // Basic SSE structure checks
    assert(encoded.find("data: ") == 0);
    assert(encoded.find("\n\n") != std::string::npos);
    assert(encoded.find("chat.completion.chunk") != std::string::npos);
    assert(encoded.find("tool_calls") != std::string::npos);
    assert(encoded.find("call_abcd1234") != std::string::npos);
    assert(encoded.find("get_weather") != std::string::npos);
    assert(encoded.find("function") != std::string::npos);
    assert(encoded.find("arguments") != std::string::npos);
    assert(encoded.find("Beijing") != std::string::npos);

    // Parse JSON from SSE envelope
    nlohmann::json j = nlohmann::json::parse(sse_to_json_str(encoded));

    // Verify full JSON structure
    assert(j["object"] == "chat.completion.chunk");
    assert(j["model"] == "gpt-4");
    assert(j["choices"][0]["finish_reason"].is_null());

    // Delta-level fields (first delta for an index)
    assert(j["choices"][0]["delta"]["role"] == "assistant");
    assert(j["choices"][0]["delta"]["content"].is_null());

    // Tool call fields
    assert(j["choices"][0]["delta"]["tool_calls"][0]["index"] == 0);
    assert(j["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_abcd1234");
    assert(j["choices"][0]["delta"]["tool_calls"][0]["type"] == "function");
    assert(j["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather");
    assert(j["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] ==
           R"({"location": "Beijing"})");

    // Verify the JSON key is not "index" inside the actual tool_calls entry...
    // (it IS index in streaming, but we just want to be sure the structure is right)
    assert(j["choices"][0]["delta"]["tool_calls"][0].contains("index"));

    std::cout << "PASSED" << std::endl;
}

// ============================================================
// Test 2: SSE encoding of multiple parallel tool calls (indices 0, 1)
// ============================================================
void test_sse_parallel_tool_calls() {
    std::cout << "Test: sse_parallel_tool_calls... " << std::flush;

    ChatCompletionsSSEEncoder encoder;

    // First tool call at index 0
    auto chunk0 = OutputChunk::ToolCallDelta(
        "call_aaaa", "get_weather", R"({"location": "Beijing"})", 0);
    chunk0.id = "test-id-2";

    // Second tool call at index 1
    auto chunk1 = OutputChunk::ToolCallDelta(
        "call_bbbb", "get_time", R"({"timezone": "UTC"})", 1);
    chunk1.id = "test-id-2";

    std::string encoded0 = encoder.encode(chunk0);
    std::string encoded1 = encoder.encode(chunk1);

    // --- Verify first tool call (index 0) ---
    nlohmann::json j0 = nlohmann::json::parse(sse_to_json_str(encoded0));
    assert(j0["choices"][0]["delta"]["tool_calls"][0]["index"] == 0);
    assert(j0["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_aaaa");
    assert(j0["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather");
    assert(j0["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] ==
           R"({"location": "Beijing"})");

    // --- Verify second tool call (index 1) ---
    nlohmann::json j1 = nlohmann::json::parse(sse_to_json_str(encoded1));
    // Index should be 1 (not 0)
    assert(j1["choices"][0]["delta"]["tool_calls"][0]["index"] == 1);
    assert(j1["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_bbbb");
    assert(j1["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_time");
    assert(j1["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] ==
           R"({"timezone": "UTC"})");
    // First delta for index 1 should also include role + content
    assert(j1["choices"][0]["delta"]["role"] == "assistant");
    assert(j1["choices"][0]["delta"]["content"].is_null());

    std::cout << "PASSED" << std::endl;
}

// ============================================================
// Test 3: SSE encoding of ToolCallFinal (finish_reason = "tool_calls")
// ============================================================
void test_sse_tool_call_finish() {
    std::cout << "Test: sse_tool_call_finish... " << std::flush;

    ChatCompletionsSSEEncoder encoder;

    // First, send a tool call delta so the encoder knows about tool calls
    auto delta = OutputChunk::ToolCallDelta(
        "call_cccc", "search", R"({"q": "hello"})", 0);
    encoder.encode(delta);

    // Then send ToolCallFinal — this should produce finish_reason "tool_calls"
    auto final_chunk = OutputChunk::ToolCallFinal(
        "call_cccc", "search", R"({"q": "hello"})", 0);
    final_chunk.id = "test-id-3";
    final_chunk.created = 1234567890;

    std::string encoded = encoder.encode(final_chunk);

    nlohmann::json j = nlohmann::json::parse(sse_to_json_str(encoded));

    // Verify finish_reason is "tool_calls"
    assert(j["object"] == "chat.completion.chunk");
    assert(j["choices"][0]["finish_reason"] == "tool_calls");

    // Verify usage fields are present
    assert(j["usage"]["prompt_tokens"] == 0);
    assert(j["usage"]["completion_tokens"] == 0);
    assert(j["usage"]["total_tokens"] == 0);

    // Delta should be an empty object
    assert(j["choices"][0]["delta"].is_object());

    std::cout << "PASSED" << std::endl;
}

// ============================================================
// Test 4: Non-streaming JSON encoder with tool_calls
// ============================================================
void test_json_tool_calls() {
    std::cout << "Test: json_tool_calls... " << std::flush;

    ChatCompletionsJSONEncoder encoder;

    // Build a tool_calls array (as the non-streaming merge would produce it)
    nlohmann::json tc;
    tc["id"] = "call_xxxx";
    tc["type"] = "function";
    tc["function"]["name"] = "get_weather";
    tc["function"]["arguments"] = R"({"location": "Beijing"})";

    nlohmann::json tool_calls = nlohmann::json::array({tc});

    // The JSON encoder reads tool_calls from chunk.obj["tool_calls"]
    auto chunk = OutputChunk::FinalText("", "gpt-4");
    chunk.id = "test-id-4";
    chunk.created = 1234567890;
    chunk.obj["tool_calls"] = tool_calls;

    std::string encoded = encoder.encode(chunk);

    // Parse and verify full JSON structure
    nlohmann::json j = nlohmann::json::parse(encoded);

    // Top-level fields
    assert(j["object"] == "chat.completion");
    assert(j["model"] == "gpt-4");

    // Choice-level: finish_reason, message
    assert(j["choices"][0]["finish_reason"] == "tool_calls");
    assert(j["choices"][0]["message"]["role"] == "assistant");
    assert(j["choices"][0]["message"]["content"].is_null());

    // Tool calls array
    assert(j["choices"][0]["message"]["tool_calls"].is_array());
    assert(j["choices"][0]["message"]["tool_calls"].size() == 1);

    auto& result_tc = j["choices"][0]["message"]["tool_calls"][0];
    assert(result_tc["id"] == "call_xxxx");
    assert(result_tc["type"] == "function");
    assert(result_tc["function"]["name"] == "get_weather");
    assert(result_tc["function"]["arguments"] == R"({"location": "Beijing"})");

    // Verify usage is present
    assert(j["usage"]["prompt_tokens"] == 0);
    assert(j["usage"]["completion_tokens"] == 0);
    assert(j["usage"]["total_tokens"] == 0);

    std::cout << "PASSED" << std::endl;
}

// ============================================================
// Test 5: Non-streaming tool call merge
//
// Simulates the server.cpp merge logic that accumulates
// ToolCallDelta chunks into a single tool_calls array.
// Specifically tests that string accumulation works correctly
// with nlohmann_json (operator+= on a json value calls
// push_back, which only works for null/array types — NOT strings).
//
// The correct approach for appending to a json string is:
//   json_val = json_val.get<std::string>() + append_value;
// ============================================================
void test_non_streaming_merge() {
    std::cout << "Test: non_streaming_merge... " << std::flush;

    // Simulate the merge logic from server.cpp (lines ~492-522), with
    // correct string accumulation (nlohmann_json += is push_back-only).
    std::map<int, nlohmann::json> tool_call_map;

    // Chunk 1: first delta for index 0 (partial arguments)
    {
        auto c = OutputChunk::ToolCallDelta(
            "call_merge1", "get_weather", R"({"location": ")", 0);
        int idx = c.tool_call_index;
        auto& tc = tool_call_map[idx];
        if (!tc.contains("id")) {
            // First chunk for this index: set metadata
            if (!c.tool_call_id.empty()) {
                tc["id"] = c.tool_call_id;
            }
            tc["type"] = "function";
            if (!c.function_name.empty()) {
                tc["function"]["name"] = c.function_name;
            }
            tc["function"]["arguments"] = "";
        }
        // Append using std::string — nlohmann_json += calls push_back
        // (would throw on strings or silently create arrays from null)
        tc["function"]["arguments"] =
            tc["function"]["arguments"].get<std::string>() + c.function_arguments;
    }

    // Chunk 2: second delta for index 0 (remaining arguments)
    {
        auto c = OutputChunk::ToolCallDelta(
            "call_merge1", "get_weather", R"(Beijing"})", 0);
        int idx = c.tool_call_index;
        auto& tc = tool_call_map[idx];
        // This is a subsequent chunk: tc already contains "id"
        // So we skip the initialization branch and just append
        if (!tc.contains("id")) {
            if (!c.tool_call_id.empty()) tc["id"] = c.tool_call_id;
            tc["type"] = "function";
            if (!c.function_name.empty()) tc["function"]["name"] = c.function_name;
            tc["function"]["arguments"] = "";
        }
        tc["function"]["arguments"] =
            tc["function"]["arguments"].get<std::string>() + c.function_arguments;
    }

    // Chunk 3: first delta for index 1 (single chunk tool call)
    {
        auto c = OutputChunk::ToolCallDelta(
            "call_merge2", "get_time", R"({"timezone": "UTC"})", 1);
        int idx = c.tool_call_index;
        auto& tc = tool_call_map[idx];
        if (!tc.contains("id")) {
            if (!c.tool_call_id.empty()) tc["id"] = c.tool_call_id;
            tc["type"] = "function";
            if (!c.function_name.empty()) tc["function"]["name"] = c.function_name;
            tc["function"]["arguments"] = "";
        }
        tc["function"]["arguments"] =
            tc["function"]["arguments"].get<std::string>() + c.function_arguments;
    }

    // Build the tool_calls array (must be in index order — std::map sorts by key)
    nlohmann::json tool_calls_arr = nlohmann::json::array();
    for (auto& [idx, tc] : tool_call_map) {
        // OpenAI non-streaming format does NOT include "index"
        tc.erase("index");
        tool_calls_arr.push_back(std::move(tc));
    }

    // Verify merged result: 2 tool calls
    assert(tool_calls_arr.size() == 2);

    // First tool call: get_weather with accumulated arguments
    assert(tool_calls_arr[0]["id"] == "call_merge1");
    assert(tool_calls_arr[0]["type"] == "function");
    assert(tool_calls_arr[0]["function"]["name"] == "get_weather");
    assert(tool_calls_arr[0]["function"]["arguments"] == R"({"location": "Beijing"})");
    // Ensure "index" was properly erased
    assert(!tool_calls_arr[0].contains("index"));

    // Second tool call: get_time
    assert(tool_calls_arr[1]["id"] == "call_merge2");
    assert(tool_calls_arr[1]["type"] == "function");
    assert(tool_calls_arr[1]["function"]["name"] == "get_time");
    assert(tool_calls_arr[1]["function"]["arguments"] == R"({"timezone": "UTC"})");
    assert(!tool_calls_arr[1].contains("index"));

    // Bonus: Feed the merged tool_calls_arr into ChatCompletionsJSONEncoder
    // to verify end-to-end roundtrip
    {
        ChatCompletionsJSONEncoder json_encoder;
        auto final_chunk = OutputChunk::FinalText("", "gpt-4");
        final_chunk.id = "merged-test";
        final_chunk.created = 1234567890;
        final_chunk.obj["tool_calls"] = tool_calls_arr;

        std::string encoded = json_encoder.encode(final_chunk);
        nlohmann::json j = nlohmann::json::parse(encoded);

        assert(j["choices"][0]["finish_reason"] == "tool_calls");
        assert(j["choices"][0]["message"]["content"].is_null());
        assert(j["choices"][0]["message"]["tool_calls"].size() == 2);
        assert(j["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather");
        assert(j["choices"][0]["message"]["tool_calls"][1]["function"]["name"] == "get_time");
        std::cout << "(json_roundtrip) " << std::flush;
    }

    std::cout << "PASSED" << std::endl;
}

// ============================================================
// Test 6: Edge cases
// ============================================================
void test_edge_cases() {
    std::cout << "Test: edge_cases... " << std::flush;

    // 6a: Empty function_arguments
    {
        ChatCompletionsSSEEncoder encoder;
        auto chunk = OutputChunk::ToolCallDelta("call_emptya", "test_func", "", 0);
        std::string encoded = encoder.encode(chunk);
        nlohmann::json j = nlohmann::json::parse(sse_to_json_str(encoded));
        assert(j["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == "");
        std::cout << "(empty_args) " << std::flush;
    }

    // 6b: Empty function_name
    {
        ChatCompletionsSSEEncoder encoder;
        auto chunk = OutputChunk::ToolCallDelta("call_emptyb", "", R"({"x": 1})", 0);
        std::string encoded = encoder.encode(chunk);
        nlohmann::json j = nlohmann::json::parse(sse_to_json_str(encoded));
        assert(j["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "");
        std::cout << "(empty_name) " << std::flush;
    }

    // 6c: Empty tool_call_id
    {
        ChatCompletionsSSEEncoder encoder;
        auto chunk = OutputChunk::ToolCallDelta("", "test_func", R"({"x": 1})", 0);
        std::string encoded = encoder.encode(chunk);
        nlohmann::json j = nlohmann::json::parse(sse_to_json_str(encoded));
        // When id is empty, the encoder should still produce something — the
        // "id" field is skipped entirely if empty (streaming first-delta branch)
        // So the tool_call should NOT have an "id" field
        assert(!j["choices"][0]["delta"]["tool_calls"][0].contains("id") ||
               j["choices"][0]["delta"]["tool_calls"][0]["id"] == "");
        std::cout << "(empty_id) " << std::flush;
    }

    // 6d: is_done still works correctly after tool calls
    {
        ChatCompletionsSSEEncoder encoder;

        // End marker
        auto end_chunk = OutputChunk::EndMarker();
        assert(encoder.is_done(end_chunk));

        // ToolCallDelta is NOT done
        auto tool_chunk = OutputChunk::ToolCallDelta("call_id", "func", "{}", 0);
        assert(!encoder.is_done(tool_chunk));

        // ToolCallFinal is NOT done
        auto final_chunk = OutputChunk::ToolCallFinal("call_id", "func", "{}", 0);
        assert(!encoder.is_done(final_chunk));

        // Regular text delta is NOT done
        auto text_chunk = OutputChunk::TextDelta("hello", "gpt-4");
        assert(!encoder.is_done(text_chunk));

        std::cout << "(is_done) " << std::flush;
    }

    // 6e: SSE done_marker unchanged after tool call support
    {
        ChatCompletionsSSEEncoder encoder;
        std::string done = encoder.done_marker();
        // Must still produce standard OpenAI [DONE] marker
        assert(done == "data: [DONE]\n\n");
        std::cout << "(done_marker) " << std::flush;
    }

    // 6f: Subsequent delta for same index (only function.arguments)
    {
        ChatCompletionsSSEEncoder encoder;

        // First delta for index 0 — has full metadata
        auto first = OutputChunk::ToolCallDelta("call_sub", "my_func", R"({"a":)", 0);
        std::string first_encoded = encoder.encode(first);
        nlohmann::json j_first = nlohmann::json::parse(sse_to_json_str(first_encoded));
        assert(j_first["choices"][0]["delta"]["tool_calls"][0].contains("id"));
        assert(j_first["choices"][0]["delta"]["tool_calls"][0].contains("function"));
        assert(j_first["choices"][0]["delta"]["tool_calls"][0]["function"].contains("name"));

        // Second delta for index 0 — should only have function.arguments
        auto second = OutputChunk::ToolCallDelta("call_sub", "my_func", R"( "b": 2})", 0);
        std::string second_encoded = encoder.encode(second);
        nlohmann::json j_second = nlohmann::json::parse(sse_to_json_str(second_encoded));
        // Should NOT have "id" or "function"."name" in subsequent delta
        assert(!j_second["choices"][0]["delta"]["tool_calls"][0].contains("id"));
        assert(!j_second["choices"][0]["delta"]["tool_calls"][0]["function"].contains("name"));
        // But should have arguments
        assert(j_second["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == R"( "b": 2})");

        std::cout << "(subsequent_delta) " << std::flush;
    }

    std::cout << "PASSED" << std::endl;
}

// ============================================================
// Test 7: Non-streaming merge with ToolCallFinal (not just ToolCallDelta)
//
// The server.cpp merge handles both ToolCallDelta and ToolCallFinal types.
// Verify that ToolCallFinal chunks are also correctly merged.
// ============================================================
void test_merge_with_tool_call_final() {
    std::cout << "Test: merge_with_tool_call_final... " << std::flush;

    std::map<int, nlohmann::json> tool_call_map;

    // Use ToolCallFinal (not delta) for the merge
    {
        auto c = OutputChunk::ToolCallFinal(
            "call_final1", "final_func", R"({"complete": true})", 0);
        int idx = c.tool_call_index;
        auto& tc = tool_call_map[idx];
        if (!tc.contains("id")) {
            if (!c.tool_call_id.empty()) tc["id"] = c.tool_call_id;
            tc["type"] = "function";
            if (!c.function_name.empty()) tc["function"]["name"] = c.function_name;
            tc["function"]["arguments"] = "";
        }
        tc["function"]["arguments"] =
            tc["function"]["arguments"].get<std::string>() + c.function_arguments;
    }

    // Build the array
    nlohmann::json tool_calls_arr = nlohmann::json::array();
    for (auto& [idx, tc] : tool_call_map) {
        tc.erase("index");
        tool_calls_arr.push_back(std::move(tc));
    }

    // Verify
    assert(tool_calls_arr.size() == 1);
    assert(tool_calls_arr[0]["id"] == "call_final1");
    assert(tool_calls_arr[0]["function"]["name"] == "final_func");
    assert(tool_calls_arr[0]["function"]["arguments"] == R"({"complete": true})");

    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "=== Tool Call Tests ===" << std::endl;

    test_sse_single_tool_call_delta();
    test_sse_parallel_tool_calls();
    test_sse_tool_call_finish();
    test_json_tool_calls();
    test_non_streaming_merge();
    test_merge_with_tool_call_final();
    test_edge_cases();

    std::cout << "\nAll tests PASSED!" << std::endl;
    return 0;
}
