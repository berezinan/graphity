"""Tests for language extractors: Java, C, C++, Ruby, C#, Kotlin, Scala, PHP, Swift, Go, Julia, Fortran, JS/TS, .NET project files, XAML."""
from __future__ import annotations
from pathlib import Path
import pytest
from graphify.extract import (
    extract_java, extract_c, extract_cpp, extract_ruby,
    extract_csharp, extract_kotlin, extract_scala, extract_php,
    extract_swift, extract_go, extract_julia, extract_js, extract_fortran,
    extract_groovy, extract_sln, extract_csproj, extract_xaml, extract_razor,
    extract_dm, extract_dmi, extract_dmm, extract_dmf,
    extract_powershell, extract_apex, extract_verilog,
    extract_bsl, extract_edt_mdo, extract_edt_rights, extract_edt_form,
    extract_edt_dcs, extract_edt_oform, extract_edt_cmi, _make_id,
    extract_powershell_manifest,
)
from graphify.extractors.bsl import (
    _edt_type_kind, _edt_ref_target, _EDT_REF_TAGS, _EDT_REF_CONTAINER_TAGS,
    _EDT_CHILD_KINDS, _EDT_SUBKIND_CONFIRMED, _EDT_SUBKIND_CONVENTION,
    _v8_elements, _oform_tree_uuids, _oform_element_names, _OFORM_TOKEN_RE,
    _EDT_KIND_PREFIXES, _EDT_PLURAL_TO_KIND, _edt_project_kind, _edt_id_scope,
    EDT_PROJECT_CONFIGURATION, EDT_PROJECT_EXTENSION, EDT_PROJECT_EXTERNAL_OBJECTS,
)

FIXTURES = Path(__file__).parent / "fixtures"

# tree-sitter-dm is an optional extra (#1104) - it ships no Linux/Mac wheel, so it
# is not installed by a default `uv sync`. Skip the .dm/.dme grammar tests when the
# grammar is absent (.dmi/.dmm/.dmf use no tree-sitter and are always tested).
import importlib.util as _ilu
_needs_dm = pytest.mark.skipif(
    _ilu.find_spec("tree_sitter_dm") is None,
    reason="tree-sitter-dm not installed (optional [dm] extra)",
)


def _labels(r):
    return [n["label"] for n in r["nodes"]]

def _relations(r):
    return {e["relation"] for e in r["edges"]}

def _calls(r):
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    return {
        (node_by_id.get(e["source"], e["source"]), node_by_id.get(e["target"], e["target"]))
        for e in r["edges"] if e["relation"] == "calls"
    }


def _references(r):
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    return [
        (
            node_by_id.get(e["source"], e["source"]),
            node_by_id.get(e["target"], e["target"]),
            e,
        )
        for e in r["edges"] if e["relation"] == "references"
    ]


def _edges_with_relation(r, *relations):
    return [e for e in r["edges"] if e["relation"] in relations]


def _normalize_symbol_label(label: str) -> str:
    return label.strip("()").lstrip(".")


def _node_by_label(result: dict, label: str) -> dict:
    for node in result["nodes"]:
        if node.get("label") == label or _normalize_symbol_label(node.get("label", "")) == label:
            return node
    raise AssertionError(f"missing node label {label!r}")


def _edge_labels(result: dict, relation: str, context: str | None = None) -> set[tuple[str, str]]:
    labels = {node["id"]: _normalize_symbol_label(node["label"]) for node in result["nodes"]}
    pairs = set()
    for edge in result["edges"]:
        if edge.get("relation") != relation:
            continue
        if context is not None and edge.get("context") != context:
            continue
        pairs.add((labels.get(edge["source"], edge["source"]), labels.get(edge["target"], edge["target"])))
    return pairs


# ── Java ──────────────────────────────────────────────────────────────────────

def test_java_no_error():
    r = extract_java(FIXTURES / "sample.java")
    assert "error" not in r

def test_java_finds_class():
    r = extract_java(FIXTURES / "sample.java")
    assert any("DataProcessor" in l for l in _labels(r))

def test_java_finds_interface():
    r = extract_java(FIXTURES / "sample.java")
    assert any("Processor" in l for l in _labels(r))

def test_java_finds_methods():
    r = extract_java(FIXTURES / "sample.java")
    labels = _labels(r)
    assert any("addItem" in l for l in labels)
    assert any("process" in l for l in labels)

def test_java_finds_imports():
    r = extract_java(FIXTURES / "sample.java")
    assert "imports" in _relations(r)


def test_java_import_edges_have_import_context():
    r = extract_java(FIXTURES / "sample.java")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)

def test_java_no_dangling_edges():
    r = extract_java(FIXTURES / "sample.java")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids


def test_java_enum_constants_have_case_of_edge():
    r = extract_java(FIXTURES / "sample.java")
    labels = _labels(r)
    assert "OK" in labels
    assert "GAME_DONE" in labels
    assert ("ErrorCode", "OK") in _edge_labels(r, "case_of")
    assert ("ErrorCode", "GAME_DONE") in _edge_labels(r, "case_of")


# ── C ────────────────────────────────────────────────────────────────────────

def test_c_no_error():
    r = extract_c(FIXTURES / "sample.c")
    assert "error" not in r

def test_c_finds_functions():
    r = extract_c(FIXTURES / "sample.c")
    labels = _labels(r)
    assert any("process" in l for l in labels)
    assert any("main" in l for l in labels)

def test_c_finds_includes():
    r = extract_c(FIXTURES / "sample.c")
    assert "imports" in _relations(r)

def test_c_emits_calls():
    r = extract_c(FIXTURES / "sample.c")
    assert any(e["relation"] == "calls" for e in r["edges"])

def test_c_calls_are_extracted():
    r = extract_c(FIXTURES / "sample.c")
    for e in r["edges"]:
        if e["relation"] == "calls":
            assert e["confidence"] == "EXTRACTED"


def test_c_import_edges_have_import_context():
    r = extract_c(FIXTURES / "sample.c")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_c_parameter_and_return_type_contexts():
    r = extract_c(FIXTURES / "sample.c")
    assert ("make_rect", "Rectangle") in _edge_labels(r, "references", "parameter_type")
    assert ("make_rect", "Rectangle") in _edge_labels(r, "references", "return_type")


def test_c_call_edges_have_call_context():
    r = extract_c(FIXTURES / "sample.c")
    call_edges = _edges_with_relation(r, "calls")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)


# ── C++ ───────────────────────────────────────────────────────────────────────

def test_cpp_no_error():
    r = extract_cpp(FIXTURES / "sample.cpp")
    assert "error" not in r

def test_cpp_finds_class():
    r = extract_cpp(FIXTURES / "sample.cpp")
    assert any("HttpClient" in l for l in _labels(r))

def test_cpp_finds_methods():
    r = extract_cpp(FIXTURES / "sample.cpp")
    labels = _labels(r)
    # C++ extractor captures the constructor and public-visible methods
    assert any("HttpClient" in l for l in labels)

def test_cpp_finds_includes():
    r = extract_cpp(FIXTURES / "sample.cpp")
    assert "imports" in _relations(r)


def test_cpp_import_edges_have_import_context():
    r = extract_cpp(FIXTURES / "sample.cpp")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_cpp_method_parameter_and_return_type_contexts():
    r = extract_cpp(FIXTURES / "sample.cpp")
    assert ("get", "string") in _edge_labels(r, "references", "parameter_type")
    assert ("get", "string") in _edge_labels(r, "references", "return_type")


def test_cpp_field_and_template_argument_contexts():
    r = extract_cpp(FIXTURES / "sample.cpp")
    assert ("HttpClient", "string") in _edge_labels(r, "references", "field")
    assert ("HttpClient", "vector") in _edge_labels(r, "references", "field")
    assert ("HttpClient", "string") in _edge_labels(r, "references", "generic_arg")


def test_cpp_class_inherits_edge():
    """Regression for #915: `class Derived : public Base {}` should emit an inherits edge."""
    r = extract_cpp(FIXTURES / "sample.cpp")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    found = any(
        "AuthedHttpClient" in node_by_id.get(e["source"], "")
        and "HttpClient" in node_by_id.get(e["target"], "")
        for e in r["edges"] if e["relation"] == "inherits"
    )
    assert found, "AuthedHttpClient should have inherits edge to HttpClient"


def test_cpp_struct_inherits_edge():
    """Structs use the same `: Base` syntax as classes and must also emit inherits."""
    r = extract_cpp(FIXTURES / "sample.cpp")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    found = any(
        "RetryingHttpClient" in node_by_id.get(e["source"], "")
        and "HttpClient" in node_by_id.get(e["target"], "")
        for e in r["edges"] if e["relation"] == "inherits"
    )
    assert found, "RetryingHttpClient (struct) should have inherits edge to HttpClient"


def test_cpp_generic_parents_include_type_argument_references():
    """`class PooledClient : public Connection<HttpClient>` must emit the inherits
    edge to Connection AND a generic_arg reference to the HttpClient type argument,
    matching the Java base-class behaviour (_emit_java_parent_type)."""
    r = extract_cpp(FIXTURES / "sample.cpp")
    assert ("PooledClient", "Connection") in _edge_labels(r, "inherits")
    assert ("PooledClient", "HttpClient") in _edge_labels(r, "references", "generic_arg")


# ── CUDA ──────────────────────────────────────────────────────────────────────
# CUDA is a C++ superset, so .cu/.cuh route through the C++ (tree-sitter-cpp)
# extractor. These tests guard that __global__/__device__ kernels, host
# functions, structs and includes are all extracted.

def test_cuda_no_error():
    r = extract_cpp(FIXTURES / "sample.cu")
    assert "error" not in r

def test_cuda_finds_kernel_and_device_functions():
    r = extract_cpp(FIXTURES / "sample.cu")
    labels = _labels(r)
    assert any("saxpy" in l for l in labels)   # __global__ kernel
    assert any("dot" in l for l in labels)     # __device__ function

def test_cuda_finds_struct():
    r = extract_cpp(FIXTURES / "sample.cu")
    assert any("Vec3" in l for l in _labels(r))

def test_cuda_finds_includes():
    r = extract_cpp(FIXTURES / "sample.cu")
    assert "imports" in _relations(r)

def test_cuda_host_call_edges():
    r = extract_cpp(FIXTURES / "sample.cu")
    calls = _calls(r)
    assert ("host_norm()", "dot()") in calls
    assert ("main()", "host_norm()") in calls


# Metal Shading Language is a C++14-derived language, so .metal files route
# through the C++ extractor just like CUDA does.

def test_metal_is_code_extension():
    from graphify.detect import CODE_EXTENSIONS
    assert ".metal" in CODE_EXTENSIONS


def test_metal_no_error():
    r = extract_cpp(FIXTURES / "sample.metal")
    assert "error" not in r


def test_metal_finds_kernel_function_and_struct():
    r = extract_cpp(FIXTURES / "sample.metal")
    labels = _labels(r)
    assert any("Vec3" in l for l in labels)
    assert any("dot3" in l for l in labels)
    assert any("saxpy" in l for l in labels)


# ── Ruby ─────────────────────────────────────────────────────────────────────

def test_ruby_no_error():
    r = extract_ruby(FIXTURES / "sample.rb")
    assert "error" not in r

def test_ruby_finds_class():
    r = extract_ruby(FIXTURES / "sample.rb")
    assert any("ApiClient" in l for l in _labels(r))

def test_ruby_finds_methods():
    r = extract_ruby(FIXTURES / "sample.rb")
    labels = _labels(r)
    assert any("get" in l for l in labels)
    assert any("post" in l for l in labels)

def test_ruby_finds_function():
    r = extract_ruby(FIXTURES / "sample.rb")
    assert any("parse_response" in l for l in _labels(r))


def test_ruby_inherits_edge():
    """`class Sub < Base` must emit an inherits edge.

    Ruby exposes the base class in the `superclass` field, but there was no
    Ruby branch in the inheritance handler, so the edge was silently dropped.
    """
    r = extract_ruby(FIXTURES / "sample.rb")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    found = any(
        "TimeoutApiClient" in node_by_id.get(e["source"], "")
        and node_by_id.get(e["target"], "") == "ApiClient"
        for e in r["edges"] if e["relation"] == "inherits"
    )
    assert found, "TimeoutApiClient should have inherits edge to ApiClient"


# ── C# ───────────────────────────────────────────────────────────────────────

def test_csharp_no_error():
    r = extract_csharp(FIXTURES / "sample.cs")
    assert "error" not in r

def test_csharp_finds_class():
    r = extract_csharp(FIXTURES / "sample.cs")
    assert any("DataProcessor" in l for l in _labels(r))

def test_csharp_finds_interface():
    r = extract_csharp(FIXTURES / "sample.cs")
    assert any("IProcessor" in l for l in _labels(r))

def test_csharp_finds_methods():
    r = extract_csharp(FIXTURES / "sample.cs")
    labels = _labels(r)
    assert any("Process" in l for l in labels)

def test_csharp_finds_usings():
    r = extract_csharp(FIXTURES / "sample.cs")
    assert "imports" in _relations(r)

def test_csharp_inherits_edge():
    r = extract_csharp(FIXTURES / "sample.cs")
    inherits = [e for e in r["edges"] if e["relation"] == "inherits"]
    assert len(inherits) >= 1

def test_csharp_implements_iprocessor():
    r = extract_csharp(FIXTURES / "sample.cs")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    found = any(
        "DataProcessor" in node_by_id.get(e["source"], "") and
        "IProcessor" in node_by_id.get(e["target"], "")
        for e in r["edges"] if e["relation"] == "implements"
    )
    assert found, "DataProcessor should have implements edge to IProcessor"


def test_csharp_splits_inherits_and_implements_edges():
    result = extract_csharp(FIXTURES / "sample.cs")
    assert ("DataProcessor", "Processor") in _edge_labels(result, "inherits")
    assert ("DataProcessor", "IProcessor") in _edge_labels(result, "implements")


def test_csharp_parameter_return_and_generic_contexts():
    result = extract_csharp(FIXTURES / "sample.cs")
    assert ("Build", "HttpClient") in _edge_labels(result, "references", "parameter_type")
    assert ("Build", "Result") in _edge_labels(result, "references", "return_type")
    assert ("Build", "DataProcessor") in _edge_labels(result, "references", "generic_arg")


def test_java_normalizes_inherits_and_implements():
    result = extract_java(FIXTURES / "sample.java")
    assert ("DataProcessor", "BaseProcessor") in _edge_labels(result, "inherits")
    assert ("DataProcessor", "Processor") in _edge_labels(result, "implements")


def test_java_generic_parents_include_type_argument_references(tmp_path):
    source = tmp_path / "GenericParents.java"
    source.write_text(
        "class Dependency {}\n"
        "interface Event {}\n"
        "class Base<T> {}\n"
        "interface Handler<T> {}\n"
        "interface DerivedHandler extends Handler<Event> {}\n"
        "class Service extends Base<Dependency> implements Handler<Event> {}\n"
    )

    result = extract_java(source)

    assert ("Service", "Base") in _edge_labels(result, "inherits")
    assert ("Service", "Handler") in _edge_labels(result, "implements")
    refs = _edge_labels(result, "references", "generic_arg")
    assert ("Service", "Dependency") in refs
    assert ("Service", "Event") in refs
    assert ("DerivedHandler", "Handler") in _edge_labels(result, "inherits")
    assert ("DerivedHandler", "Event") in refs


def test_java_type_parameters_do_not_emit_references(tmp_path):
    source = tmp_path / "TypeParameters.java"
    source.write_text(
        "class Payload {}\n"
        "class Base<X> {}\n"
        "class Box<T> extends Base<T> {\n"
        "    T value;\n"
        "    List<T> values;\n"
        "    <U> U convert(T input, List<U> mapped, List<Payload> retained) {\n"
        "        return null;\n"
        "    }\n"
        "    <V> Box(V value) {}\n"
        "}\n"
    )

    result = extract_java(source)

    references = _references(result)
    assert not [edge for _, target, edge in references if target in {"T", "U", "V"}]
    assert not [
        node
        for node in result["nodes"]
        if node.get("label") in {"T", "U", "V"} and not node.get("source_file")
    ]
    assert ("Box", "Base") in _edge_labels(result, "inherits")
    assert ("convert", "Payload") in _edge_labels(result, "references", "generic_arg")


def test_java_parameter_return_generic_and_attribute_contexts():
    result = extract_java(FIXTURES / "sample.java")
    assert ("build", "HttpClient") in _edge_labels(result, "references", "parameter_type")
    assert ("build", "Result") in _edge_labels(result, "references", "return_type")
    assert ("build", "DataProcessor") in _edge_labels(result, "references", "generic_arg")
    assert ("build", "Override") in _edge_labels(result, "references", "attribute")


def test_java_field_type_references_have_field_context(tmp_path):
    source = tmp_path / "Fields.java"
    source.write_text(
        "class PaymentGateway {}\n"
        "class Handler {}\n"
        "class CheckoutService {\n"
        "    PaymentGateway gateway;\n"
        "    List<Handler> handlers;\n"
        "}\n"
    )
    result = extract_java(source)
    assert ("CheckoutService", "PaymentGateway") in _edge_labels(
        result, "references", "field"
    )
    assert ("CheckoutService", "Handler") in _edge_labels(
        result, "references", "generic_arg"
    )


def test_java_record_component_type_references(tmp_path):
    source = tmp_path / "RecordComponents.java"
    source.write_text(
        "class Payload {}\n"
        "class Item {}\n"
        "class Attachment {}\n"
        "record Order(Payload payload, List<Item> items, int count, "
        "Attachment... attachments) {}\n"
    )

    result = extract_java(source)

    assert ("Order", "Payload") in _edge_labels(result, "references", "field")
    # `List` is a java.util library type: skipped as noise, so only its user-type
    # generic argument (`Item`) survives, not the container itself.
    assert ("Order", "List") not in _edge_labels(result, "references")
    assert ("Order", "Item") in _edge_labels(result, "references", "generic_arg")
    assert ("Order", "Attachment") in _edge_labels(result, "references", "field")


def test_java_record_components_skip_type_parameters(tmp_path):
    source = tmp_path / "GenericRecord.java"
    source.write_text(
        "class Payload {}\n"
        "class Box<X> {}\n"
        "record Batch<T>(T value, Box<T> boxed, Box<Payload> retained) {}\n"
    )

    result = extract_java(source)

    assert ("Batch", "T") not in _edge_labels(result, "references")
    assert not [
        node
        for node in result["nodes"]
        if node.get("label") == "T" and not node.get("source_file")
    ]
    assert ("Batch", "Box") in _edge_labels(result, "references", "field")
    assert ("Batch", "Payload") in _edge_labels(result, "references", "generic_arg")


def test_java_type_annotations_have_attribute_context(tmp_path):
    source = tmp_path / "TypeAnnotations.java"
    source.write_text(
        '@Service\n'
        '@Entity(name = "checkout")\n'
        'class CheckoutService {}\n'
    )

    result = extract_java(source)

    refs = _edge_labels(result, "references", "attribute")
    assert ("CheckoutService", "Service") in refs
    assert ("CheckoutService", "Entity") in refs


def test_java_enum_and_annotation_declarations_are_type_nodes(tmp_path):
    source = tmp_path / "TypeDeclarations.java"
    source.write_text(
        "enum PaymentStatus { PENDING, PAID }\n"
        "@interface Audited {}\n"
        "class Order { PaymentStatus status; }\n"
        "@Audited class CheckoutService {}\n"
    )

    result = extract_java(source)

    assert ("TypeDeclarations.java", "PaymentStatus") in _edge_labels(
        result, "contains"
    )
    assert ("TypeDeclarations.java", "Audited") in _edge_labels(result, "contains")
    assert ("Order", "PaymentStatus") in _edge_labels(
        result, "references", "field"
    )
    assert ("CheckoutService", "Audited") in _edge_labels(
        result, "references", "attribute"
    )
    definitions = {
        node["label"]: node
        for node in result["nodes"]
        if node.get("label") in {"PaymentStatus", "Audited"}
    }
    assert definitions["PaymentStatus"].get("source_file") == str(source)
    assert definitions["Audited"].get("source_file") == str(source)


def test_csharp_field_type_references_have_field_context():
    r = extract_csharp(FIXTURES / "sample.cs")
    refs = _references(r)
    assert any(
        "DataProcessor" in src and "HttpClient" in tgt and edge.get("context") == "field"
        for src, tgt, edge in refs
    ), "DataProcessor field declarations should reference HttpClient with field context"


def test_csharp_property_type_references_have_field_context():
    r = extract_csharp(FIXTURES / "sample.cs")
    field_refs = _edge_labels(r, "references", "field")
    # `public Processor Owner { get; set; }` — property type -> field ref.
    assert ("DataProcessor", "Processor") in field_refs
    # `public List<Processor> Workers { get; set; }` — the List container -> field.
    assert ("DataProcessor", "List") in field_refs
    # ...and the generic argument -> generic_arg.
    assert ("DataProcessor", "Processor") in _edge_labels(r, "references", "generic_arg")


def test_csharp_call_edges_have_call_context():
    r = extract_csharp(FIXTURES / "sample.cs")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    assert any(
        "Process" in node_by_id.get(e["source"], "")
        and "Validate" in node_by_id.get(e["target"], "")
        and e.get("context") == "call"
        for e in r["edges"] if e["relation"] == "calls"
    ), "C# call edges should retain call context"


def test_csharp_import_edges_have_import_context():
    r = extract_csharp(FIXTURES / "sample.cs")
    import_edges = [e for e in r["edges"] if e["relation"] == "imports"]
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


# ── Kotlin ───────────────────────────────────────────────────────────────────

def test_kotlin_no_error():
    r = extract_kotlin(FIXTURES / "sample.kt")
    assert "error" not in r

def test_kotlin_finds_class():
    r = extract_kotlin(FIXTURES / "sample.kt")
    assert any("HttpClient" in l for l in _labels(r))

def test_kotlin_finds_data_class():
    r = extract_kotlin(FIXTURES / "sample.kt")
    assert any("Config" in l for l in _labels(r))

def test_kotlin_finds_methods():
    r = extract_kotlin(FIXTURES / "sample.kt")
    labels = _labels(r)
    assert any("get" in l for l in labels)
    assert any("post" in l for l in labels)

def test_kotlin_finds_function():
    r = extract_kotlin(FIXTURES / "sample.kt")
    assert any("createClient" in l for l in _labels(r))

def test_kotlin_enum_entries_have_case_of_edge():
    # #1700 (Kotlin half): enum entries must be nodes with case_of edges to the enum.
    r = extract_kotlin(FIXTURES / "sample.kt")
    labels = _labels(r)
    assert "NORMAL" in labels and "GROUP" in labels and "SYSTEM" in labels
    assert ("ChatType", "NORMAL") in _edge_labels(r, "case_of")
    assert ("ChatType", "SYSTEM") in _edge_labels(r, "case_of")

def test_kotlin_emits_in_file_calls():
    """Regression test for the call-walker `simple_identifier` /
    `identifier` rename — see graphify-kmp's PythonParityTest."""
    r = extract_kotlin(FIXTURES / "sample.kt")
    calls = _calls(r)
    # In sample.kt: get() and post() both call buildRequest(), and
    # createClient() invokes Config and HttpClient (constructor calls).
    assert (".get()", ".buildRequest()") in calls
    assert (".post()", ".buildRequest()") in calls
    assert ("createClient()", "Config") in calls
    assert ("createClient()", "HttpClient") in calls


def test_kotlin_splits_inherits_and_implements():
    r = extract_kotlin(FIXTURES / "sample.kt")
    assert ("DataProcessor", "BaseProcessor") in _edge_labels(r, "inherits")
    assert ("DataProcessor", "Loggable") in _edge_labels(r, "implements")


def test_kotlin_interface_delegation_emits_implements():
    """`class Foo : Bar by baz` wraps the delegated interface in an
    `explicit_delegation` node — it must still emit an implements edge."""
    r = extract_kotlin(FIXTURES / "sample.kt")
    assert ("LoggingList", "MutableList") in _edge_labels(r, "implements")


def test_kotlin_parameter_return_generic_and_field_contexts():
    r = extract_kotlin(FIXTURES / "sample.kt")
    assert ("run", "DataProcessor") in _edge_labels(r, "references", "parameter_type")
    assert ("run", "Result") in _edge_labels(r, "references", "return_type")
    assert ("run", "DataProcessor") in _edge_labels(r, "references", "generic_arg")
    assert ("DataProcessor", "Result") in _edge_labels(r, "references", "field")

def test_kotlin_builtin_types_not_emitted_as_references():
    # kotlin.* scalar/collection/core types used as parameter, return, or field
    # types carry no useful graph meaning: they never resolve to a project node,
    # so emitting `references` edges to them is pure noise (mirrors the Java
    # _JAVA_BUILTIN_TYPES / Python _PYTHON_ANNOTATION_NOISE handling).
    r = extract_kotlin(FIXTURES / "sample.kt")
    ref_targets = {target for (_, target) in _edge_labels(r, "references")}
    for builtin in ("String", "Int"):
        assert builtin not in ref_targets, (
            f"builtin type {builtin!r} should not be a references target"
        )

def test_kotlin_user_types_still_emit_references():
    # Guard against over-filtering: a user-defined class sharing its name with a
    # common domain-modeling identifier (Result) must still resolve to a real
    # edge - the builtin filter is a fixed name list, so it must stay narrow
    # enough not to swallow common user-chosen names like a sealed-class "Result".
    r = extract_kotlin(FIXTURES / "sample.kt")
    assert ("DataProcessor", "Result") in _edge_labels(r, "references", "field")
    assert ("run", "DataProcessor") in _edge_labels(r, "references", "parameter_type")


# ── Scala ─────────────────────────────────────────────────────────────────────

def test_scala_no_error():
    r = extract_scala(FIXTURES / "sample.scala")
    assert "error" not in r

def test_scala_finds_class():
    r = extract_scala(FIXTURES / "sample.scala")
    assert any("HttpClient" in l for l in _labels(r))

def test_scala_finds_object():
    r = extract_scala(FIXTURES / "sample.scala")
    assert any("HttpClientFactory" in l for l in _labels(r))

def test_scala_finds_methods():
    r = extract_scala(FIXTURES / "sample.scala")
    labels = _labels(r)
    assert any("get" in l for l in labels)
    assert any("post" in l for l in labels)


def test_scala_import_edges_have_import_context():
    r = extract_scala(FIXTURES / "sample.scala")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_scala_splits_inherits_and_mixes_in():
    r = extract_scala(FIXTURES / "sample.scala")
    assert ("HttpClient", "BaseClient") in _edge_labels(r, "inherits")
    assert ("HttpClient", "Loggable") in _edge_labels(r, "mixes_in")


def test_scala_constructor_parameter_field_context():
    r = extract_scala(FIXTURES / "sample.scala")
    assert ("HttpClient", "Config") in _edge_labels(r, "references", "field")


def test_scala_val_definition_field_context():
    r = extract_scala(FIXTURES / "sample.scala")
    assert ("HttpClient", "Config") in _edge_labels(r, "references", "field")


def test_scala_var_definition_field_context():
    r = extract_scala(FIXTURES / "sample.scala")
    assert ("HttpClient", "BaseClient") in _edge_labels(r, "references", "field")


def test_scala_method_return_type_context():
    r = extract_scala(FIXTURES / "sample.scala")
    assert ("create", "HttpClient") in _edge_labels(r, "references", "return_type")


def test_scala_call_edges_have_call_context():
    r = extract_scala(FIXTURES / "sample.scala")
    call_edges = _edges_with_relation(r, "calls")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)


# ── PHP ───────────────────────────────────────────────────────────────────────

def test_php_no_error():
    r = extract_php(FIXTURES / "sample.php")
    assert "error" not in r

def test_php_finds_class():
    r = extract_php(FIXTURES / "sample.php")
    assert any("ApiClient" in l for l in _labels(r))

def test_php_finds_methods():
    r = extract_php(FIXTURES / "sample.php")
    labels = _labels(r)
    assert any("get" in l for l in labels)
    assert any("post" in l for l in labels)

def test_php_finds_function():
    r = extract_php(FIXTURES / "sample.php")
    assert any("parseResponse" in l for l in _labels(r))

def test_php_finds_imports():
    r = extract_php(FIXTURES / "sample.php")
    assert "imports" in _relations(r)


def test_php_import_edges_have_import_context():
    r = extract_php(FIXTURES / "sample.php")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_php_call_edges_have_call_context():
    r = extract_php(FIXTURES / "sample.php")
    call_edges = _edges_with_relation(r, "calls")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)

def test_php_finds_static_property_access():
    r = extract_php(FIXTURES / "sample_php_static_prop.php")
    assert "uses_static_prop" in _relations(r)

def test_php_static_prop_target_is_holding_class():
    r = extract_php(FIXTURES / "sample_php_static_prop.php")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    uses_prop = [
        (node_by_id.get(e["source"], e["source"]), node_by_id.get(e["target"], e["target"]))
        for e in r["edges"] if e["relation"] == "uses_static_prop"
    ]
    assert any("DefaultPalette" in tgt for _, tgt in uses_prop)

def test_php_finds_config_helper_call():
    r = extract_php(FIXTURES / "sample_php_config.php")
    assert "uses_config" in _relations(r)

def test_php_config_helper_target_matches_first_segment():
    r = extract_php(FIXTURES / "sample_php_config.php")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    uses_cfg = [
        (node_by_id.get(e["source"], e["source"]), node_by_id.get(e["target"], e["target"]))
        for e in r["edges"] if e["relation"] == "uses_config"
    ]
    assert any("Throttle" in tgt for _, tgt in uses_cfg)

def test_php_finds_container_bind():
    r = extract_php(FIXTURES / "sample_php_container.php")
    assert "bound_to" in _relations(r)

def test_php_container_bind_links_contract_to_implementation():
    r = extract_php(FIXTURES / "sample_php_container.php")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    bound = [
        (node_by_id.get(e["source"], e["source"]), node_by_id.get(e["target"], e["target"]))
        for e in r["edges"] if e["relation"] == "bound_to"
    ]
    assert any("PaymentGateway" in src and "StripeGateway" in tgt for src, tgt in bound)

def test_php_finds_event_listeners():
    r = extract_php(FIXTURES / "sample_php_listen.php")
    assert "listened_by" in _relations(r)

def test_php_event_listener_links_event_to_listener():
    r = extract_php(FIXTURES / "sample_php_listen.php")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    listened = [
        (node_by_id.get(e["source"], e["source"]), node_by_id.get(e["target"], e["target"]))
        for e in r["edges"] if e["relation"] == "listened_by"
    ]
    assert any("UserRegistered" in src and "SendWelcomeEmail" in tgt for src, tgt in listened)


def test_php_splits_inherits_implements_mixes_in():
    r = extract_php(FIXTURES / "sample.php")
    assert ("DataProcessor", "BaseProcessor") in _edge_labels(r, "inherits")
    assert ("DataProcessor", "Loggable") in _edge_labels(r, "implements")
    assert ("DataProcessor", "HasName") in _edge_labels(r, "mixes_in")


def test_php_property_parameter_and_return_contexts():
    r = extract_php(FIXTURES / "sample.php")
    assert ("DataProcessor", "Result") in _edge_labels(r, "references", "field")
    assert ("run", "DataProcessor") in _edge_labels(r, "references", "parameter_type")
    assert ("run", "Result") in _edge_labels(r, "references", "return_type")


def test_php_constructor_property_promotion_contexts():
    # PHP 8 constructor property promotion: a promoted param is both a
    # constructor parameter (parameter_type) and a class field (field).
    r = extract_php(FIXTURES / "sample.php")
    assert ("Service", "Result") in _edge_labels(r, "references", "field")
    assert ("__construct", "Result") in _edge_labels(r, "references", "parameter_type")
    # A non-promoted param must not leak a field edge onto the class.
    assert ("Service", "string") not in _edge_labels(r, "references", "field")


# ── Swift ────────────────────────────────────────────────────────────────────

def test_swift_no_error():
    r = extract_swift(FIXTURES / "sample.swift")
    assert "error" not in r

def test_swift_finds_class():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("DataProcessor" in l for l in _labels(r))

def test_swift_finds_protocol():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("Processor" in l for l in _labels(r))

def test_swift_finds_struct():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("Config" in l for l in _labels(r))

def test_swift_finds_methods():
    r = extract_swift(FIXTURES / "sample.swift")
    labels = _labels(r)
    assert any("addItem" in l for l in labels)
    assert any("process" in l for l in labels)

def test_swift_finds_function():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("createProcessor" in l for l in _labels(r))

def test_swift_finds_imports():
    r = extract_swift(FIXTURES / "sample.swift")
    assert "imports" in _relations(r)


def test_swift_import_edges_have_import_context():
    r = extract_swift(FIXTURES / "sample.swift")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)

def test_swift_no_dangling_edges():
    r = extract_swift(FIXTURES / "sample.swift")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids
        # #1327: targets must resolve to a node too, else build.py prunes the edge.
        assert e["target"] in node_ids, f"dangling target {e['target']} ({e['relation']})"


def test_swift_imports_survive_build():
    # #1327: `import Foundation` / `import UIKit` previously emitted edges to bare
    # module ids with no backing node, so build.py dropped 100% of Swift imports.
    from graphify.build import build_from_json
    r = extract_swift(FIXTURES / "sample.swift")
    import_edges = [e for e in r["edges"] if e["relation"] == "imports"]
    assert import_edges, "extractor should emit Swift import edges"
    node_ids = {n["id"] for n in r["nodes"]}
    for e in import_edges:
        assert e["target"] in node_ids  # synthesized module node exists
    # Imported modules are tagged type=module (anchor nodes, #1327/#1330).
    module_labels = {n["label"] for n in r["nodes"] if n.get("type") == "module"}
    assert {"Foundation", "UIKit"} <= module_labels
    # No private bookkeeping key should leak into output edges.
    assert all("_import_label" not in e for e in r["edges"])
    # Edges must survive the build (which prunes edges with unknown endpoints).
    G = build_from_json(r)
    surviving = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "imports"
    ]
    assert surviving, "Swift import edges must survive build_from_json (#1327)"

def test_swift_finds_actor():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("CacheManager" in l for l in _labels(r))

def test_swift_finds_enum():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("NetworkError" in l for l in _labels(r))

def test_swift_finds_enum_methods():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("describe" in l for l in _labels(r))

def test_swift_finds_enum_cases():
    r = extract_swift(FIXTURES / "sample.swift")
    labels = _labels(r)
    assert any("timeout" in l for l in labels)
    assert any("connectionFailed" in l for l in labels)

def test_swift_enum_cases_have_case_of_edge():
    r = extract_swift(FIXTURES / "sample.swift")
    case_edges = [e for e in r["edges"] if e["relation"] == "case_of"]
    assert len(case_edges) >= 2

def test_swift_enum_associated_value_type_emits_references():
    r = extract_swift(FIXTURES / "sample.swift")
    assert ("NetworkError", "Config") in _edge_labels(r, "references", "type")

def test_swift_finds_deinit():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("deinit" in l for l in _labels(r))

def test_swift_finds_subscript():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("subscript" in l for l in _labels(r))

def test_swift_extension_methods_attach_to_type():
    r = extract_swift(FIXTURES / "sample.swift")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    method_edges = [e for e in r["edges"] if e["relation"] == "method"]
    found = False
    for e in method_edges:
        src_label = node_by_id.get(e["source"], "")
        tgt_label = node_by_id.get(e["target"], "")
        if "Config" in src_label and "isValid" in tgt_label:
            found = True
            break
    assert found, "extension method isValid should attach to Config"

def test_swift_extension_does_not_duplicate_type_node():
    r = extract_swift(FIXTURES / "sample.swift")
    config_nodes = [n for n in r["nodes"] if n["label"] == "Config"]
    assert len(config_nodes) == 1, f"Config should appear once, got {len(config_nodes)}"

def test_swift_protocol_conformance_emits_implements():
    r = extract_swift(FIXTURES / "sample.swift")
    assert ("DataProcessor", "Processor") in _edge_labels(r, "implements")


def test_swift_extension_conformance_emits_implements():
    r = extract_swift(FIXTURES / "sample.swift")
    assert ("DataProcessor", "Loggable") in _edge_labels(r, "implements")


def test_swift_splits_inherits_and_implements():
    r = extract_swift(FIXTURES / "sample.swift")
    assert ("DataProcessor", "BaseProcessor") in _edge_labels(r, "inherits")
    assert ("DataProcessor", "Processor") in _edge_labels(r, "implements")


def test_swift_parameter_return_generic_and_field_contexts():
    r = extract_swift(FIXTURES / "sample.swift")
    assert ("run", "DataProcessor") in _edge_labels(r, "references", "parameter_type")
    assert ("run", "Result") in _edge_labels(r, "references", "return_type")
    assert ("run", "DataProcessor") in _edge_labels(r, "references", "generic_arg")
    assert ("DataProcessor", "Result") in _edge_labels(r, "references", "field")

def test_swift_emits_calls():
    r = extract_swift(FIXTURES / "sample.swift")
    calls = _calls(r)
    assert any("process" in src and "validate" in tgt for src, tgt in calls)

def test_swift_call_edges_have_call_context():
    r = extract_swift(FIXTURES / "sample.swift")
    call_edges = _edges_with_relation(r, "calls")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)


def test_swift_extension_across_files_merges_into_canonical_type():
    """`extension Foo` in a separate file from `class Foo` must resolve to a
    single Foo node. tree-sitter-swift parses both as `class_declaration` and
    node ids carry the file stem, so without a corpus-level merge each file
    would emit its own Foo."""
    from graphify.extract import extract
    paths = sorted((FIXTURES / "swift_cross_file").glob("*.swift"))
    r = extract(paths, cache_root=Path("/tmp/graphify-test-no-cache"))
    foo_nodes = [n for n in r["nodes"] if n["label"] == "Foo"]
    assert len(foo_nodes) == 1, f"Foo should appear once, got {len(foo_nodes)}: {[n['id'] for n in foo_nodes]}"
    foo_id = foo_nodes[0]["id"]
    method_targets = {
        e["target"] for e in r["edges"]
        if e["relation"] == "method" and e["source"] == foo_id
    }
    method_labels = {n["label"] for n in r["nodes"] if n["id"] in method_targets}
    assert any("one" in l for l in method_labels), f"one() should attach to Foo, got {method_labels}"
    assert any("two" in l for l in method_labels), f"extension method two() should attach to Foo, got {method_labels}"


# ── Elixir ────────────────────────────────────────────────────────────────────

from graphify.extract import extract_elixir

def test_elixir_finds_module():
    r = extract_elixir(FIXTURES / "sample.ex")
    assert "error" not in r
    labels = [n["label"] for n in r["nodes"]]
    assert any("MyApp.Accounts.User" in l for l in labels)

def test_elixir_finds_functions():
    r = extract_elixir(FIXTURES / "sample.ex")
    labels = [n["label"] for n in r["nodes"]]
    assert any("create" in l for l in labels)
    assert any("find" in l for l in labels)
    assert any("validate" in l for l in labels)

def test_elixir_finds_imports():
    r = extract_elixir(FIXTURES / "sample.ex")
    import_edges = [e for e in r["edges"] if e["relation"] == "imports"]
    assert len(import_edges) >= 2


def test_elixir_import_edges_have_import_context():
    r = extract_elixir(FIXTURES / "sample.ex")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_elixir_multi_alias_expands():
    """`alias Foo.{Bar, Baz}` must emit one imports edge per expanded module.

    The brace form is a `dot` node with a trailing `tuple`; the single-alias
    handler only matched a bare `alias` child, so every multi-alias import was
    silently dropped.
    """
    r = extract_elixir(FIXTURES / "sample.ex")
    import_segs = [
        e["target"].rsplit("_", 1)[-1]
        for e in r["edges"] if e["relation"] == "imports"
    ]
    # from `alias MyApp.Schemas.{Account, Token}`
    assert "account" in import_segs, "MyApp.Schemas.Account import missing"
    assert "token" in import_segs, "MyApp.Schemas.Token import missing"

def test_elixir_finds_calls():
    r = extract_elixir(FIXTURES / "sample.ex")
    calls = {(e["source"], e["target"]) for e in r["edges"] if e["relation"] == "calls"}
    labels = {n["id"]: n["label"] for n in r["nodes"]}
    assert any("create" in labels.get(src, "") and "validate" in labels.get(tgt, "") for src, tgt in calls)


def test_elixir_call_edges_have_call_context():
    r = extract_elixir(FIXTURES / "sample.ex")
    call_edges = _edges_with_relation(r, "calls")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)

def test_elixir_method_edges():
    r = extract_elixir(FIXTURES / "sample.ex")
    methods = [e for e in r["edges"] if e["relation"] == "method"]
    assert len(methods) >= 3


# ── Objective-C ──────────────────────────────────────────────────────────────
from graphify.extract import extract_objc


def test_objc_finds_interface():
    r = extract_objc(FIXTURES / "sample.m")
    labels = [n["label"] for n in r["nodes"]]
    assert "Animal" in labels


def test_objc_finds_subclass():
    r = extract_objc(FIXTURES / "sample.m")
    labels = [n["label"] for n in r["nodes"]]
    assert "Dog" in labels


def test_objc_finds_methods():
    r = extract_objc(FIXTURES / "sample.m")
    labels = [n["label"] for n in r["nodes"]]
    assert any("speak" in l or "fetch" in l or "initWithName" in l for l in labels)


def test_objc_finds_imports():
    r = extract_objc(FIXTURES / "sample.m")
    import_edges = [e for e in r["edges"] if e["relation"] == "imports"]
    assert len(import_edges) >= 1


def test_objc_import_edges_have_import_context():
    r = extract_objc(FIXTURES / "sample.m")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_objc_inherits_edge():
    r = extract_objc(FIXTURES / "sample.m")
    inherits = [e for e in r["edges"] if e["relation"] == "inherits"]
    assert len(inherits) >= 1


def test_objc_splits_inherits_and_implements():
    r = extract_objc(FIXTURES / "sample.m")
    assert ("Animal", "NSObject") in _edge_labels(r, "inherits")
    assert ("Dog", "Animal") in _edge_labels(r, "inherits")
    assert ("Animal", "SampleDelegate") in _edge_labels(r, "implements")


def test_objc_protocol_adopts_protocol():
    """`@protocol Derived <Base>` must emit an implements edge Derived->Base.
    Protocol-on-protocol adoption nests under a protocol_reference_list node
    (distinct from the parameterized_arguments node used by @interface
    adoption), so the edge was previously dropped. Protocol nodes are labeled
    `<Name>`, so the edge reads (<Derived>, <Base>)."""
    r = extract_objc(FIXTURES / "sample.m")
    assert ("<Derived>", "<Base>") in _edge_labels(r, "implements")


def test_objc_property_type_context():
    r = extract_objc(FIXTURES / "sample.m")
    assert ("Animal", "NSString") in _edge_labels(r, "references", "field")


def test_objc_no_dangling_edges():
    r = extract_objc(FIXTURES / "sample.m")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"Dangling source: {e}"


def test_objc_resolves_self_method_calls():
    """`[self speak]` inside Dog.fetch must produce a calls edge. The method-body
    second pass was dead code for ObjC because the grammar emits a simple selector
    as `identifier`, not `selector`/`keyword_argument_list` (#1475)."""
    r = extract_objc(FIXTURES / "sample.m")
    nid2label = {n["id"]: n["label"] for n in r["nodes"]}
    calls = [nid2label.get(e["target"]) for e in r["edges"] if e["relation"] == "calls"]
    assert any(t and "speak" in t for t in calls), calls


def test_objc_class_method_labeled_with_plus(tmp_path):
    """`+ (…)shared` is a class method and must be labeled +shared, not -shared (#1475)."""
    p = tmp_path / "S.m"
    p.write_text("@implementation S\n+ (instancetype)shared { return nil; }\n- (void)go { }\n@end\n")
    labels = {n["label"] for n in extract_objc(p)["nodes"]}
    assert "+shared" in labels and "-go" in labels


def test_objc_compound_selector_call_resolves(tmp_path):
    """A compound message `[self a:x b:y]` resolves to the compound method def (#1475)."""
    p = tmp_path / "V.m"
    p.write_text(
        "@implementation V\n"
        "- (void)tableView:(id)tv numberOfRowsInSection:(int)s { }\n"
        "- (void)go { [self tableView:nil numberOfRowsInSection:0]; }\n"
        "@end\n"
    )
    r = extract_objc(p)
    nid2label = {n["id"]: n["label"] for n in r["nodes"]}
    calls = [nid2label.get(e["target"]) for e in r["edges"] if e["relation"] == "calls"]
    assert any(t and "tableViewnumberOfRowsInSection" in t for t in calls), calls


def test_objc_generic_property_type_extracted(tmp_path):
    """`NSArray<Product *> *` must reference the element type Product (and the
    container NSArray); the generic wrapper made the type invisible before (#1475)."""
    p = tmp_path / "M.h"
    p.write_text("@interface M : NSObject\n@property (strong) NSArray<Product *> *items;\n@end\n")
    refs = _edge_labels(extract_objc(p), "references", "field")
    assert ("M", "Product") in refs
    assert ("M", "NSArray") in refs


def test_objc_module_import_edge(tmp_path):
    """`@import Foundation;` / `@import UIKit.UIView;` produce imports edges (#1475)."""
    from graphify.extract import _make_id
    p = tmp_path / "X.m"
    p.write_text("@import Foundation;\n@import UIKit.UIView;\n@implementation X\n@end\n")
    targets = {e["target"] for e in extract_objc(p)["edges"] if e["relation"] == "imports"}
    assert _make_id("Foundation") in targets and _make_id("UIKit") in targets


def test_objc_header_dispatch_routes_objc_not_c(tmp_path):
    """An ObjC `.h` (has @interface) routes to extract_objc; a plain C `.h` stays
    on extract_c, so C/C++ headers are never hijacked by the sniff (#1475)."""
    from graphify.extract import _get_extractor, extract_objc as _eo, extract_c as _ec
    objc_h = tmp_path / "AppDelegate.h"
    objc_h.write_text("@interface AppDelegate : NSObject <UIApplicationDelegate>\n@end\n")
    c_h = tmp_path / "util.h"
    c_h.write_text("#include <stdio.h>\nint add(int a, int b);\nstruct Point { int x; };\n")
    assert _get_extractor(objc_h) is _eo
    assert _get_extractor(c_h) is _ec


def test_objc_ns_assume_nonnull_macro_does_not_break_parsing(tmp_path):
    """`NS_ASSUME_NONNULL_BEGIN` before `@interface` made tree-sitter-objc fail to
    emit a class_interface node, swallowing the whole interface; blanking the
    argument-less macro restores it (#1475)."""
    p = tmp_path / "AlertManager.h"
    p.write_text(
        "#import <Foundation/Foundation.h>\n"
        "NS_ASSUME_NONNULL_BEGIN\n"
        "@class Other;\n"
        "@interface AlertManager : NSObject\n"
        "- (void)show;\n"
        "@end\n"
        "NS_ASSUME_NONNULL_END\n"
    )
    r = extract_objc(p)
    labels = {n["label"] for n in r["nodes"]}
    assert "AlertManager" in labels
    assert ("AlertManager", "NSObject") in _edge_labels(r, "inherits")
    # `@class Other;` is only a forward declaration; it must not mint a class node.
    assert "Other" not in labels


def test_objc_macro_free_header_unchanged(tmp_path):
    """A macro-free header still parses exactly as before (regression)."""
    p = tmp_path / "Plain.h"
    p.write_text(
        "@interface Plain : NSObject\n"
        "- (void)go;\n"
        "@end\n"
    )
    r = extract_objc(p)
    labels = {n["label"] for n in r["nodes"]}
    assert "Plain" in labels
    assert ("Plain", "NSObject") in _edge_labels(r, "inherits")


def test_objc_quoted_import_edges_resolve_to_real_nodes(tmp_path):
    """Quoted `#import "X.h"` edges must target the real (disambiguated) file node id,
    not the bare stem, which gets salted away when a `.h`/`.m` pair exists and left
    the import edge dangling (#1475)."""
    from graphify.extract import extract
    (tmp_path / "Product.h").write_text("@interface Product : NSObject\n@end\n")
    (tmp_path / "Product.m").write_text("#import \"Product.h\"\n@implementation Product\n@end\n")
    (tmp_path / "Order.h").write_text("@interface Order : NSObject\n@end\n")
    (tmp_path / "Order.m").write_text("#import \"Order.h\"\n@implementation Order\n@end\n")
    consumer_a = tmp_path / "ConsumerA.m"
    consumer_a.write_text("#import \"Product.h\"\n@implementation ConsumerA\n@end\n")
    consumer_b = tmp_path / "ConsumerB.m"
    consumer_b.write_text("#import \"Order.h\"\n@implementation ConsumerB\n@end\n")
    files = [
        tmp_path / "Product.h", tmp_path / "Product.m",
        tmp_path / "Order.h", tmp_path / "Order.m",
        consumer_a, consumer_b,
    ]
    r = extract(files, parallel=False)
    node_ids = {n["id"] for n in r["nodes"]}
    id_to_label = {n["id"]: n.get("label", "") for n in r["nodes"]}
    import_edges = [e for e in r["edges"] if e["relation"] in ("imports", "imports_from")]
    assert import_edges
    for e in import_edges:
        # No dangling targets...
        assert e["target"] in node_ids, f"dangling import target: {e}"
        # ...and no self-loops: a `.m` importing its own `.h` must resolve to the
        # header file node, not get salted back to the importing `.m` (#1475).
        assert e["source"] != e["target"], f"self-loop import edge: {e}"
        # every quoted import targets a header (.h) file node
        assert str(id_to_label.get(e["target"], "")).endswith(".h"), (
            f"import target is not a header file node: {e} -> {id_to_label.get(e['target'])}"
        )
    # the self-import (Product.m -> Product.h) specifically lands on the .h variant
    prod_imports = [e for e in import_edges if id_to_label.get(e["source"], "").endswith("Product.m")]
    assert prod_imports and all(id_to_label.get(e["target"]) == "Product.h" for e in prod_imports), (
        f"Product.m should import the Product.h node, got {[(id_to_label.get(e['source']), id_to_label.get(e['target'])) for e in prod_imports]}"
    )


def test_objc_alloc_init_emits_type_reference(tmp_path):
    """`[[Foo alloc] init]` must emit a `references` edge to the project class Foo (#1475)."""
    from graphify.extract import extract
    (tmp_path / "Foo.h").write_text("@interface Foo : NSObject\n@end\n")
    (tmp_path / "Foo.m").write_text("#import \"Foo.h\"\n@implementation Foo\n@end\n")
    user = tmp_path / "User.m"
    user.write_text(
        "#import \"Foo.h\"\n"
        "@implementation User\n"
        "- (void)build { Foo *x = [[Foo alloc] init]; }\n"
        "@end\n"
    )
    r = extract([tmp_path / "Foo.h", tmp_path / "Foo.m", user], parallel=False)
    assert ("-build", "Foo") in _edge_labels(r, "references")


def test_objc_alloc_init_unknown_class_no_resolved_edge(tmp_path):
    """`[[Unknown alloc] init]` with no such class must not produce a resolved
    reference edge (the sourceless stub is collapsed only when a real class exists)."""
    p = tmp_path / "Caller.m"
    p.write_text(
        "@implementation Caller\n"
        "- (void)build { id x = [[Unknown alloc] init]; }\n"
        "- (void)other { [self build]; [x doStuff]; }\n"
        "@end\n"
    )
    r = extract_objc(p)
    # The single-file extractor emits the edge to a sourceless stub; assert there is
    # no resolved reference to a *real* (sourced) Unknown node and that ordinary
    # selector sends ([self build] / [x doStuff]) produce no alloc reference.
    sourced_ids = {n["id"] for n in r["nodes"] if n.get("source_file")}
    refs = [e for e in r["edges"] if e["relation"] == "references"]
    for e in refs:
        assert e["target"] not in sourced_ids, f"unexpected resolved ref: {e}"


def test_objc_dot_syntax_property_accesses_edge(tmp_path):
    """self.name dot-syntax resolves to an accesses edge within the same class."""
    p = tmp_path / "Dog.m"
    p.write_text(
        "@implementation Dog\n"
        "- (NSString *)name { return @\"Rex\"; }\n"
        "- (void)greet { NSLog(@\"%@\", self.name); }\n"
        "@end\n"
    )
    r = extract_objc(p)
    accesses = [(e["source"], e["target"]) for e in r["edges"]
                if e["relation"] == "accesses"]
    nid2label = {n["id"]: n["label"] for n in r["nodes"]}
    assert len(accesses) == 1
    assert nid2label[accesses[0][1]] == "-name"


def test_objc_dot_syntax_no_fanout_two_same_named_properties(tmp_path):
    """Two classes each declaring -name: self.name in A must NOT fan out to B's -name."""
    p = tmp_path / "AB.m"
    p.write_text(
        "@implementation A\n"
        "- (NSString *)name { return @\"A\"; }\n"
        "- (void)show { NSLog(@\"%@\", self.name); }\n"
        "@end\n"
        "@implementation B\n"
        "- (NSString *)name { return @\"B\"; }\n"
        "- (void)show { NSLog(@\"%@\", self.name); }\n"
        "@end\n"
    )
    r = extract_objc(p)
    accesses = [e for e in r["edges"] if e["relation"] == "accesses"]
    assert len(accesses) == 2, f"expected 2 scoped accesses, got {len(accesses)}: {accesses}"
    nid2label = {n["id"]: n["label"] for n in r["nodes"]}
    for e in accesses:
        src_label = nid2label[e["source"]]
        tgt_label = nid2label[e["target"]]
        assert src_label == "-show" and tgt_label == "-name"


def test_objc_dot_syntax_unresolvable_property_zero_edges(tmp_path):
    """Accessing a property not defined in the current class produces zero accesses edges."""
    p = tmp_path / "X.m"
    p.write_text(
        "@implementation X\n"
        "- (void)run { NSLog(@\"%@\", self.missing); }\n"
        "@end\n"
    )
    r = extract_objc(p)
    accesses = [e for e in r["edges"] if e["relation"] == "accesses"]
    assert len(accesses) == 0


def test_objc_selector_expression_calls_edge(tmp_path):
    """@selector(uniqueMethod) with exactly one match produces a calls edge."""
    p = tmp_path / "Sched.m"
    p.write_text(
        "@implementation Sched\n"
        "- (void)fetch { }\n"
        "- (void)schedule { [self performSelector:@selector(fetch)]; }\n"
        "@end\n"
    )
    r = extract_objc(p)
    nid2label = {n["id"]: n["label"] for n in r["nodes"]}
    sel_calls = [(nid2label.get(e["source"]), nid2label.get(e["target"]))
                 for e in r["edges"]
                 if e["relation"] == "calls" and e.get("context") == "call"]
    assert ("-schedule", "-fetch") in sel_calls


def test_objc_selector_no_fanout_two_same_named_methods(tmp_path):
    """@selector(doThing) with two doThing methods must emit zero calls edges."""
    p = tmp_path / "Dual.m"
    p.write_text(
        "@implementation A\n"
        "- (void)doThing { }\n"
        "- (void)run { [self performSelector:@selector(doThing)]; }\n"
        "@end\n"
        "@implementation B\n"
        "- (void)doThing { }\n"
        "@end\n"
    )
    r = extract_objc(p)
    nid2label = {n["id"]: n["label"] for n in r["nodes"]}
    sel_edges = [e for e in r["edges"]
                 if e["relation"] == "calls"
                 and nid2label.get(e["target"], "").endswith("doThing")]
    assert len(sel_edges) == 0, f"expected 0 selector edges with ambiguous name, got {sel_edges}"


def test_objc_dot_syntax_substring_sibling_exact_match(tmp_path):
    """A substring-colliding sibling must neither be falsely matched nor suppress
    the real match: `self.name` with both `-name` and `-surname` present resolves
    to `-name` ONLY (exact id, not a `endswith` suffix) (#1475)."""
    p = tmp_path / "Person.m"
    p.write_text(
        "@implementation Person\n"
        "- (NSString *)name { return @\"n\"; }\n"
        "- (NSString *)surname { return @\"s\"; }\n"
        "- (void)show { NSLog(@\"%@\", self.name); }\n"
        "@end\n"
    )
    r = extract_objc(p)
    nid2label = {n["id"]: n["label"] for n in r["nodes"]}
    accesses = [(nid2label.get(e["source"]), nid2label.get(e["target"]))
                for e in r["edges"] if e["relation"] == "accesses"]
    assert ("-show", "-name") in accesses, f"self.name must resolve to -name: {accesses}"
    assert ("-show", "-surname") not in accesses, f"self.name must NOT match -surname: {accesses}"


def test_objc_selector_substring_method_exact_match(tmp_path):
    """@selector(doThing) must resolve to `-doThing` exactly, not be suppressed by
    a substring-colliding `-reallyDoThing` (exact match, not suffix) (#1475)."""
    p = tmp_path / "Worker.m"
    p.write_text(
        "@implementation Worker\n"
        "- (void)doThing { }\n"
        "- (void)reallyDoThing { }\n"
        "- (void)run { [self performSelector:@selector(doThing)]; }\n"
        "@end\n"
    )
    r = extract_objc(p)
    nid2label = {n["id"]: n["label"] for n in r["nodes"]}
    sel_calls = [(nid2label.get(e["source"]), nid2label.get(e["target"]))
                 for e in r["edges"]
                 if e["relation"] == "calls" and e.get("context") == "call"]
    assert ("-run", "-doThing") in sel_calls, f"@selector(doThing) must resolve to -doThing: {sel_calls}"
    assert ("-run", "-reallyDoThing") not in sel_calls


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

def test_go_receiver_methods_share_type_node():
    """Methods on the same receiver type must share one canonical type node."""
    r = extract_go(FIXTURES / "sample.go")
    server_nodes = [n for n in r["nodes"] if n["label"] == "Server"]
    # Both Start() and Stop() are on *Server — should produce exactly one Server node
    assert len(server_nodes) == 1

def test_go_receiver_uses_pkg_scope():
    """Type node id should be scoped to directory, not file stem."""
    r = extract_go(FIXTURES / "sample.go")
    server_nodes = [n for n in r["nodes"] if n["label"] == "Server"]
    assert server_nodes
    # Should NOT contain the file stem "sample" in the type node id
    assert "sample" not in server_nodes[0]["id"].split(":")[0]


# ---------------------------------------------------------------------------
# Julia
# ---------------------------------------------------------------------------

def test_julia_finds_module():
    r = extract_julia(FIXTURES / "sample.jl")
    labels = [n["label"] for n in r["nodes"]]
    assert "Geometry" in labels


def test_julia_finds_structs():
    r = extract_julia(FIXTURES / "sample.jl")
    labels = [n["label"] for n in r["nodes"]]
    assert "Point" in labels
    assert "Circle" in labels


def test_julia_finds_abstract_type():
    r = extract_julia(FIXTURES / "sample.jl")
    labels = [n["label"] for n in r["nodes"]]
    assert "Shape" in labels


def test_julia_finds_functions():
    r = extract_julia(FIXTURES / "sample.jl")
    labels = [n["label"] for n in r["nodes"]]
    assert any("area" in l for l in labels)
    assert any("distance" in l for l in labels)


def test_julia_finds_short_function():
    r = extract_julia(FIXTURES / "sample.jl")
    labels = [n["label"] for n in r["nodes"]]
    assert any("perimeter" in l for l in labels)


def test_julia_finds_imports():
    r = extract_julia(FIXTURES / "sample.jl")
    import_edges = [e for e in r["edges"] if e["relation"] == "imports"]
    assert len(import_edges) >= 1


def test_julia_import_edges_have_import_context():
    r = extract_julia(FIXTURES / "sample.jl")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_julia_qualified_and_relative_imports():
    """Qualified (`using Base.Threads`) and relative (`using ..Mod`) imports
    must emit edges.

    The handler only matched bare identifiers, so scoped_identifier and
    import_path forms — and the scoped package of a selected_import — were
    silently dropped.
    """
    r = extract_julia(FIXTURES / "sample.jl")
    targets = [e["target"] for e in r["edges"] if e["relation"] == "imports"]
    assert any("base_threads" in t for t in targets), "qualified import Base.Threads missing"
    assert any("parentmodule" in t for t in targets), "relative import ParentModule missing"


def test_julia_finds_inherits():
    r = extract_julia(FIXTURES / "sample.jl")
    inherits = [e for e in r["edges"] if e["relation"] == "inherits"]
    assert len(inherits) >= 1


def test_julia_abstract_concrete_hierarchy_inherits():
    r = extract_julia(FIXTURES / "sample.jl")
    assert ("Point", "Shape") in _edge_labels(r, "inherits")
    assert ("Circle", "Shape") in _edge_labels(r, "inherits")


def test_julia_struct_field_type_context():
    r = extract_julia(FIXTURES / "sample.jl")
    assert ("Point", "Float64") in _edge_labels(r, "references", "field")
    assert ("Circle", "Point") in _edge_labels(r, "references", "field")
    assert ("Circle", "Float64") in _edge_labels(r, "references", "field")


def test_julia_finds_calls():
    r = extract_julia(FIXTURES / "sample.jl")
    call_edges = [e for e in r["edges"] if e["relation"] == "calls"]
    assert len(call_edges) >= 1


def test_julia_call_edges_have_call_context():
    r = extract_julia(FIXTURES / "sample.jl")
    call_edges = _edges_with_relation(r, "calls")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)


def test_julia_no_dangling_edges():
    r = extract_julia(FIXTURES / "sample.jl")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"Dangling source: {e}"


# ── Fortran extractor ────────────────────────────────────────────────────────

def test_fortran_finds_module():
    r = extract_fortran(FIXTURES / "sample.f90")
    assert "error" not in r
    labels = [n["label"] for n in r["nodes"]]
    assert "geometry" in labels


def test_fortran_finds_subroutines():
    r = extract_fortran(FIXTURES / "sample.f90")
    labels = [n["label"] for n in r["nodes"]]
    assert any("circle_area" in l for l in labels)
    assert any("print_area" in l for l in labels)


def test_fortran_finds_function():
    r = extract_fortran(FIXTURES / "sample.f90")
    labels = [n["label"] for n in r["nodes"]]
    assert any("distance" in l for l in labels)


def test_fortran_finds_program():
    r = extract_fortran(FIXTURES / "sample.f90")
    labels = [n["label"] for n in r["nodes"]]
    assert "main" in labels


def test_fortran_finds_use_imports():
    r = extract_fortran(FIXTURES / "sample.f90")
    import_edges = [e for e in r["edges"] if e["relation"] == "imports"]
    assert len(import_edges) >= 2


def test_fortran_use_edges_have_use_context():
    r = extract_fortran(FIXTURES / "sample.f90")
    import_edges = [e for e in r["edges"] if e["relation"] == "imports"]
    assert all(e.get("context") == "use" for e in import_edges)


def test_fortran_finds_calls():
    r = extract_fortran(FIXTURES / "sample.f90")
    call_edges = [e for e in r["edges"] if e["relation"] == "calls"]
    assert len(call_edges) >= 1


def test_fortran_finds_function_call():
    """`y = f(x)` function invocations must emit a calls edge.

    Function calls are `call_expression` (not `subroutine_call`); that node was
    never handled, so every function-to-function call was dropped. The callee is
    resolved against defined procedures so array indexing (`arr(i)`) can't
    fabricate a spurious edge.
    """
    r = extract_fortran(FIXTURES / "sample.f90")
    labels = {n["id"]: n["label"] for n in r["nodes"]}
    found = any(
        "report" in labels.get(e["source"], "")
        and "double_val" in labels.get(e["target"], "")
        for e in r["edges"] if e["relation"] == "calls"
    )
    assert found, "report() should have a calls edge to double_val()"


def test_fortran_case_insensitive_names():
    r = extract_fortran(FIXTURES / "sample.f90")
    labels = [n["label"] for n in r["nodes"]]
    assert all(l == l.lower() or "(" in l for l in labels if l.endswith(("()", "")) and not "." in l)
    assert "geometry" in labels
    assert "main" in labels


def test_fortran_finds_derived_type():
    r = extract_fortran(FIXTURES / "sample.f90")
    labels = [n["label"] for n in r["nodes"]]
    assert "point" in labels


def test_fortran_parameter_and_return_type_contexts():
    r = extract_fortran(FIXTURES / "sample.f90")
    assert ("translate", "point") in _edge_labels(r, "references", "parameter_type")
    assert ("origin", "point") in _edge_labels(r, "references", "return_type")


def test_fortran_no_dangling_edges():
    r = extract_fortran(FIXTURES / "sample.f90")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"Dangling source: {e}"


def test_fortran_capital_F_parses_preprocessed():
    r = extract_fortran(FIXTURES / "sample_preprocessed.F90")
    assert "error" not in r
    labels = [n["label"] for n in r["nodes"]]
    assert "shapes" in labels
    assert any("compute_volume" in l for l in labels)


# ── PowerShell ───────────────────────────────────────────────────────────────

def test_powershell_no_error():
    r = extract_powershell(FIXTURES / "sample.ps1")
    assert "error" not in r


def test_powershell_psm1_dispatched_and_extracted(tmp_path):
    # #1315: .psm1 modules were never indexed — no dispatch entry, no CODE_EXTENSIONS.
    from graphify.extract import _get_extractor
    mod = tmp_path / "Utils.psm1"
    mod.write_text(
        "function Get-Greeting { param([string]$Name) return \"Hi $Name\" }\n",
        encoding="utf-8",
    )
    assert _get_extractor(mod) is extract_powershell
    r = extract_powershell(mod)
    assert "error" not in r
    assert any("Get-Greeting" in n["label"] for n in r["nodes"])


def test_powershell_finds_class_and_method():
    r = extract_powershell(FIXTURES / "sample.ps1")
    labels = [n["label"] for n in r["nodes"]]
    assert "DataProcessor" in labels
    assert any("Transform" in l for l in labels)


def test_powershell_class_base_type_emits_inherits_edge():
    # `class Circle : Shape` — the base type after ':' was previously dropped
    # because the handler only read the first simple_name (the class name).
    r = extract_powershell(FIXTURES / "sample.ps1")
    assert ("Circle", "Shape") in _edge_labels(r, "inherits")


def test_powershell_property_field_type_context():
    r = extract_powershell(FIXTURES / "sample.ps1")
    assert ("DataProcessor", "string") in _edge_labels(r, "references", "field")


def test_powershell_method_parameter_and_return_type_contexts():
    r = extract_powershell(FIXTURES / "sample.ps1")
    assert ("Transform", "string") in _edge_labels(r, "references", "parameter_type")
    assert ("Transform", "string") in _edge_labels(r, "references", "return_type")
    assert ("Save", "void") in _edge_labels(r, "references", "return_type")


# ── PowerShell: Import-Module + dot-source (#1331) ───────────────────────────

def test_powershell_import_module_emits_edge():
    """Import-Module Foo at top level emits an imports_from edge."""
    r = extract_powershell(FIXTURES / "sample_import.ps1")
    assert "error" not in r
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert any("foo" in t for t in targets), f"Missing Import-Module Foo edge; targets={targets}"


def test_powershell_import_module_with_name_param():
    """Import-Module -Name Bar.psm1 resolves to module stem 'bar'."""
    r = extract_powershell(FIXTURES / "sample_import.ps1")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert any("bar" in t for t in targets), f"Missing Import-Module -Name Bar edge; targets={targets}"


def test_powershell_dot_source_forward_slash_emits_edge():
    """Dot-source `. ./Shared.psm1` emits an imports_from edge."""
    r = extract_powershell(FIXTURES / "sample_import.ps1")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert any("shared" in t for t in targets), f"Missing dot-source Shared edge; targets={targets}"


def test_powershell_dot_source_backslash_emits_edge():
    """Dot-source `. .\\Utils.ps1` (backslash path) emits an imports_from edge."""
    r = extract_powershell(FIXTURES / "sample_import.ps1")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert any("utils" in t for t in targets), f"Missing dot-source Utils edge; targets={targets}"


def test_powershell_import_module_inside_function_emits_edge():
    """Import-Module inside a function body still produces an imports_from edge."""
    r = extract_powershell(FIXTURES / "sample_import.ps1")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert any("innermod" in t for t in targets), (
        f"Missing Import-Module InnerMod edge from function body; targets={targets}"
    )


def test_powershell_import_module_not_a_raw_call():
    """Import-Module must not appear in raw_calls (it is an import, not a function call)."""
    r = extract_powershell(FIXTURES / "sample_import.ps1")
    import_module_calls = [
        rc for rc in r.get("raw_calls", [])
        if rc.get("callee", "").lower() == "import-module"
    ]
    assert not import_module_calls, (
        f"Import-Module appeared in raw_calls but should be emitted as import edge: {import_module_calls}"
    )


def test_powershell_dot_source_inside_function_emits_edge():
    """Dot-source inside a function body still produces an imports_from edge."""
    r = extract_powershell(FIXTURES / "sample_import.ps1")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert any("innershared" in t for t in targets), (
        f"Missing dot-source InnerShared edge from function body; targets={targets}"
    )


# ── PowerShell manifest (.psd1) (#1331) ──────────────────────────────────────

def test_powershell_psd1_dispatched():
    """_get_extractor should route .psd1 to extract_powershell_manifest."""
    from graphify.extract import _get_extractor
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".psd1", delete=False) as f:
        f.write(b"@{ RootModule = 'X.psm1' }")
        path = f.name
    try:
        assert _get_extractor(Path(path)) is extract_powershell_manifest
    finally:
        os.unlink(path)


def test_powershell_psd1_no_error():
    r = extract_powershell_manifest(FIXTURES / "sample.psd1")
    assert "error" not in r


def test_powershell_psd1_has_file_node():
    r = extract_powershell_manifest(FIXTURES / "sample.psd1")
    assert any("sample.psd1" in n["label"] for n in r["nodes"]), (
        f"Missing file node for sample.psd1; nodes={[n['label'] for n in r['nodes']]}"
    )


def test_powershell_psd1_root_module():
    """RootModule = 'MyModule.psm1' produces an imports_from edge to 'mymodule'."""
    r = extract_powershell_manifest(FIXTURES / "sample.psd1")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert any("mymodule" in t for t in targets), (
        f"Missing RootModule edge for MyModule; targets={targets}"
    )


def test_powershell_psd1_nested_modules():
    """NestedModules = @('Helpers.psm1', 'Logger.psm1') produces edges for both."""
    r = extract_powershell_manifest(FIXTURES / "sample.psd1")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert any("helpers" in t for t in targets), f"Missing NestedModules Helpers edge; targets={targets}"
    assert any("logger" in t for t in targets), f"Missing NestedModules Logger edge; targets={targets}"


def test_powershell_psd1_required_modules_string():
    """RequiredModules string form 'PSReadLine' produces an imports_from edge."""
    r = extract_powershell_manifest(FIXTURES / "sample.psd1")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert any("psreadline" in t for t in targets), (
        f"Missing RequiredModules PSReadLine edge; targets={targets}"
    )


def test_powershell_psd1_required_modules_hashtable():
    """RequiredModules hashtable form @{{ ModuleName='Pester' }} produces an imports_from edge."""
    r = extract_powershell_manifest(FIXTURES / "sample.psd1")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert any("pester" in t for t in targets), (
        f"Missing RequiredModules Pester (hashtable form) edge; targets={targets}"
    )


def test_powershell_psd1_no_moduleversion_as_edge():
    """ModuleVersion values ('5.0', '1.0.0') must NOT appear as import targets."""
    r = extract_powershell_manifest(FIXTURES / "sample.psd1")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert not any(t in targets for t in ("5_0", "1_0_0", "5.0", "1.0.0")), (
        f"ModuleVersion string leaked into import targets: {targets}"
    )


def test_powershell_psd1_no_dangling_edges():
    """All imports_from edge sources must exist in the node set."""
    r = extract_powershell_manifest(FIXTURES / "sample.psd1")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"Dangling source in edge: {e}"


# ── TypeScript dynamic imports ───────────────────────────────────────────────

def test_ts_dynamic_import_no_error():
    r = extract_js(FIXTURES / "dynamic_import.ts")
    assert "error" not in r

def test_ts_dynamic_import_extracts_edges():
    """Dynamic import() calls inside functions should produce imports_from edges."""
    r = extract_js(FIXTURES / "dynamic_import.ts")
    dyn_edges = [e for e in r["edges"] if e["relation"] == "imports_from"]
    targets = {e["target"] for e in dyn_edges}
    # Should find: static ./logger, dynamic ./mayaEngine.js, dynamic ./queue.js
    assert any("logger" in t for t in targets), f"Missing static import of logger: {targets}"
    assert any("mayaengine" in t.lower() for t in targets), f"Missing dynamic import of mayaEngine: {targets}"
    assert any("queue" in t.lower() for t in targets), f"Missing dynamic import of queue: {targets}"

def test_ts_dynamic_import_confidence():
    """Dynamic imports should have EXTRACTED confidence (they are deterministic string literals)."""
    r = extract_js(FIXTURES / "dynamic_import.ts")
    dyn_edges = [e for e in r["edges"]
                 if e["relation"] == "imports_from"
                 and "mayaengine" in e["target"].lower()]
    assert len(dyn_edges) >= 1
    assert dyn_edges[0]["confidence"] == "EXTRACTED"

def test_ts_dynamic_import_source_is_function():
    """Dynamic import edge source should be the enclosing function, not the file."""
    r = extract_js(FIXTURES / "dynamic_import.ts")
    node_labels = {n["id"]: n["label"] for n in r["nodes"]}
    dyn_edges = [e for e in r["edges"]
                 if e["relation"] == "imports_from"
                 and "mayaengine" in e["target"].lower()]
    assert len(dyn_edges) >= 1
    src_label = node_labels.get(dyn_edges[0]["source"], "")
    assert "processInbound" in src_label, f"Expected processInbound as source, got {src_label}"

def test_ts_no_dynamic_import_in_sync_fn():
    """Functions without dynamic imports should not get spurious imports_from edges."""
    r = extract_js(FIXTURES / "dynamic_import.ts")
    node_ids = {n["label"]: n["id"] for n in r["nodes"]}
    sync_nid = node_ids.get("syncOnly()")
    if sync_nid:
        sync_imports = [e for e in r["edges"]
                        if e["source"] == sync_nid and e["relation"] == "imports_from"]
        assert len(sync_imports) == 0

def test_ts_dynamic_template_literal_skipped():
    """Dynamic template literals (with ${}) must not produce an imports_from edge."""
    r = extract_js(FIXTURES / "dynamic_import.ts")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    # loadHandler uses `./handlers/${handlerName}` — no static path, must be absent
    assert not any("handler" in t.lower() and "$" in t for t in targets), \
        f"Garbage edge from dynamic template literal found: {targets}"
    # More robust: no target should contain a brace character
    assert not any("{" in t or "}" in t for t in targets), \
        f"Target contains unresolved template expression: {targets}"

def test_ts_static_template_literal_resolved():
    """Static template literals (no ${}) should resolve the same as a plain string."""
    r = extract_js(FIXTURES / "dynamic_import.ts")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert any("statichelper" in t.lower() for t in targets), \
        f"Static template literal import not resolved: {targets}"


def test_js_local_const_does_not_emit_phantom_node(tmp_path):
    """Local const/let/var inside an arrow callback must NOT emit a node (#1077).

    Previously `_js_extra_walk` recursed into arrow_function bodies and
    emitted a node for every `const x = ...` inside e.g. `describe(() => {})`,
    so bare names like `set`, `sorted` collided across unrelated test files.
    """
    src = (
        "describe('suite', () => {\n"
        "  const inner = new Set([1, 2, 3]);\n"
        "  let other = [1, 2];\n"
        "});\n"
        "\n"
        "const moduleConst = new Set([4, 5]);\n"
        "export const exportedConst = { a: 1 };\n"
    )
    f = tmp_path / "scope_guard.js"
    f.write_text(src)
    r = extract_js(f)
    labels = _labels(r)

    # Locals inside the arrow callback must not produce nodes.
    assert "inner" not in labels, f"phantom node for arrow-body local 'inner': {labels}"
    assert "other" not in labels, f"phantom node for arrow-body local 'other': {labels}"

    # Module-level consts should still produce nodes.
    assert "moduleConst" in labels, f"module-level const 'moduleConst' missing: {labels}"
    assert "exportedConst" in labels, f"exported const 'exportedConst' missing: {labels}"


def test_js_module_level_arrow_produces_node_and_call_edges(tmp_path):
    """Module-level arrow functions must still emit a node and capture their calls (#1077).

    The scope guard must not accidentally suppress top-level arrow functions.
    """
    src = (
        "function helper() { return 1; }\n"
        "const handler = () => {\n"
        "  helper();\n"
        "};\n"
    )
    f = tmp_path / "arrows.js"
    f.write_text(src)
    r = extract_js(f)
    labels = _labels(r)
    relations = _relations(r)

    assert any("handler" in l for l in labels), f"module-level arrow 'handler' missing: {labels}"
    assert "calls" in relations, f"expected 'calls' edge from handler->helper: {relations}"


def test_ts_local_const_does_not_emit_phantom_node(tmp_path):
    """Scope guard applies to TypeScript files too (shared _js_extra_walk path)."""
    src = (
        "describe('suite', () => {\n"
        "  const inner: Set<number> = new Set([1, 2]);\n"
        "});\n"
        "\n"
        "export const topLevel = { a: 1 };\n"
    )
    f = tmp_path / "scope_guard.ts"
    f.write_text(src)
    r = extract_js(f)
    labels = _labels(r)

    assert "inner" not in labels, f"phantom TS node for arrow-body local 'inner': {labels}"
    assert "topLevel" in labels, f"module-level TS const 'topLevel' missing: {labels}"


def test_ts_constructor_injection_calls_edge(tmp_path):
    """this.repo.findById() in a class with constructor(private repo: IUserRepository)
    must produce a calls edge from getUser() to findById() (#1316)."""
    from graphify.extract import extract
    repo_ts = tmp_path / "repo.ts"
    repo_ts.write_text(
        "export interface IUserRepository {\n"
        "  findById(id: string): Promise<any>;\n"
        "  save(user: any): Promise<void>;\n"
        "}\n"
    )
    svc_ts = tmp_path / "service.ts"
    svc_ts.write_text(
        "import { IUserRepository } from './repo';\n"
        "\n"
        "export class UserService {\n"
        "  constructor(private repo: IUserRepository) {}\n"
        "\n"
        "  getUser(id: string) {\n"
        "    return this.repo.findById(id);\n"
        "  }\n"
        "}\n"
    )
    r = extract([repo_ts, svc_ts], cache_root=tmp_path / "cache")
    edge_triples = {
        (e["source"], e["relation"], e["target"])
        for e in r["edges"]
    }
    labels_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    label_triples = {
        (labels_by_id.get(s, s), rel, labels_by_id.get(t, t))
        for s, rel, t in edge_triples
    }
    calls_from_get_user = [
        (s, rel, t) for s, rel, t in label_triples
        if "getUser" in s and rel == "calls"
    ]
    assert any("findById" in t for _, _, t in calls_from_get_user), (
        f"expected getUser()->findById() calls edge, got: {calls_from_get_user}"
    )


def test_ts_this_field_receiver_not_same_file_collision(tmp_path):
    """this.db.query() should NOT match an unrelated query() in the same file (#1316)."""
    f = tmp_path / "collision.ts"
    f.write_text(
        "function query() { return 'global'; }\n"
        "\n"
        "export class Service {\n"
        "  constructor(private db: Database) {}\n"
        "\n"
        "  run() {\n"
        "    return this.db.query();\n"
        "  }\n"
        "}\n"
    )
    r = extract_js(f)
    calls_edges = [
        e for e in r["edges"]
        if e["relation"] == "calls"
    ]
    caller_labels = {n["id"]: n["label"] for n in r["nodes"]}
    run_to_query = [
        e for e in calls_edges
        if "run" in caller_labels.get(e["source"], "")
        and "query" in caller_labels.get(e["target"], "")
    ]
    assert len(run_to_query) == 0, (
        f"this.db.query() should NOT resolve to bare query() in same file: {run_to_query}"
    )


def _ts_label_calls(r, src_sub):
    labels = {n["id"]: n["label"] for n in r["nodes"]}
    return [
        labels.get(e["target"], e["target"])
        for e in r["edges"]
        if e["relation"] == "calls" and src_sub in labels.get(e["source"], e["source"])
    ]


def test_ts_injected_field_resolves_to_typed_class_not_same_named_collision(tmp_path):
    """The decisive #1316 guardrail: two classes each define `query`, but the
    injected field is typed `Database`, so `this.db.query()` must resolve to
    Database.query ONLY — never HttpClient.query (no global name-match fan-out)."""
    from graphify.extract import extract
    (tmp_path / "database.ts").write_text(
        "export class Database {\n  query(sql: string) { return sql; }\n}\n"
    )
    (tmp_path / "http.ts").write_text(
        "export class HttpClient {\n  query(url: string) { return url; }\n}\n"
    )
    (tmp_path / "service.ts").write_text(
        "import { Database } from './database';\n"
        "export class Service {\n"
        "  constructor(private db: Database) {}\n"
        "  run() { return this.db.query('x'); }\n"
        "}\n"
    )
    r = extract(
        [tmp_path / "database.ts", tmp_path / "http.ts", tmp_path / "service.ts"],
        cache_root=tmp_path / "cache",
    )
    labels = {n["id"]: n["label"] for n in r["nodes"]}
    # Find the run()->query calls edge and confirm its target is owned by Database.
    method_owner = {
        e["target"]: e["source"]
        for e in r["edges"] if e["relation"] == "method"
    }
    run_query_targets = [
        e["target"] for e in r["edges"]
        if e["relation"] == "calls"
        and "run" in labels.get(e["source"], "")
        and "query" in labels.get(e["target"], "")
    ]
    assert run_query_targets, "expected this.db.query() to resolve to a query method"
    for tgt in run_query_targets:
        owner = method_owner.get(tgt)
        assert owner is not None and labels.get(owner) == "Database", (
            f"this.db.query() must resolve to Database.query, got owner {labels.get(owner)}"
        )


def test_ts_injected_field_ambiguous_type_emits_no_edge(tmp_path):
    """If the injected field's type name is ambiguous (two classes named Database),
    the god-node guard bails — no calls edge rather than a guess (#1316)."""
    from graphify.extract import extract
    (tmp_path / "a" ).mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "database.ts").write_text(
        "export class Database {\n  query(sql: string) { return sql; }\n}\n"
    )
    (tmp_path / "b" / "database.ts").write_text(
        "export class Database {\n  query(sql: string) { return sql; }\n}\n"
    )
    (tmp_path / "service.ts").write_text(
        "export class Service {\n"
        "  constructor(private db: Database) {}\n"
        "  run() { return this.db.query('x'); }\n"
        "}\n"
    )
    r = extract(sorted(tmp_path.rglob("*.ts")), cache_root=tmp_path / "cache")
    # `query` resolution must bail (2 Database defs) -> no run()->query calls edge.
    assert not [t for t in _ts_label_calls(r, "run") if "query" in t], (
        "ambiguous Database type must not produce a this.db.query() edge"
    )


# ── Markdown ─────────────────────────────────────────────────────────────────

from graphify.extract import extract_markdown

def test_markdown_no_error():
    r = extract_markdown(FIXTURES / "deploy_guide.md")
    assert "error" not in r

def test_markdown_finds_headings():
    r = extract_markdown(FIXTURES / "deploy_guide.md")
    labels = _labels(r)
    assert any("Deploy Guide" in l for l in labels)
    assert any("Prerequisites" in l for l in labels)
    assert any("Full Deploy" in l for l in labels)
    assert any("Rollback" in l for l in labels)

def test_markdown_finds_nested_heading():
    """### Database Migration is nested under ## Full Deploy."""
    r = extract_markdown(FIXTURES / "deploy_guide.md")
    labels = _labels(r)
    assert any("Database Migration" in l for l in labels)

def test_markdown_skips_fenced_code_blocks():
    """Fenced code blocks should NOT emit nodes (#1077).

    They were always orphans (single contains edge to parent doc) and
    inflated the disconnected-component count. We still skip over their
    *contents* when parsing so the inside of a fence is not misread as a
    heading.
    """
    r = extract_markdown(FIXTURES / "deploy_guide.md")
    labels = _labels(r)
    assert not any(l.startswith("code:") for l in labels), \
        f"Expected no code:* nodes after #1077 fix, got: {[l for l in labels if l.startswith('code:')]}"

def test_markdown_contains_edges():
    """Headings should be connected via 'contains' edges (file->h, h->h)."""
    r = extract_markdown(FIXTURES / "deploy_guide.md")
    assert "contains" in _relations(r)
    contains_edges = [e for e in r["edges"] if e["relation"] == "contains"]
    # deploy_guide.md has: file->h1, h1->h2(Prerequisites), h1->h2(Full Deploy),
    # h2(Full Deploy)->h3(Database Migration), h1->h2(Rollback) = 5 edges
    assert len(contains_edges) >= 5, f"expected >= 5 contains edges, got {len(contains_edges)}"


def test_markdown_fenced_heading_not_parsed():
    """A '## heading' inside a fenced block must not produce a heading node (#1077).

    The fence-toggle skips over fenced contents so interior markdown syntax
    is not misread as document structure.
    """
    import tempfile, os
    src = (
        "# Real Heading\n"
        "\n"
        "```bash\n"
        "## Not A Heading\n"
        "echo hello\n"
        "```\n"
        "\n"
        "## Another Real Heading\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as fh:
        fh.write(src)
        fpath = fh.name
    try:
        r = extract_markdown(Path(fpath))
        labels = _labels(r)
    finally:
        os.unlink(fpath)

    assert any("Real Heading" in l for l in labels), f"'Real Heading' missing: {labels}"
    assert any("Another Real Heading" in l for l in labels), f"'Another Real Heading' missing: {labels}"
    assert not any("Not A Heading" in l for l in labels), \
        f"fenced '## Not A Heading' was incorrectly parsed as a node: {labels}"

def test_markdown_no_dangling_edges():
    r = extract_markdown(FIXTURES / "deploy_guide.md")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"Dangling source: {e}"


def _md_link_fixture(tmp_path):
    """A hub doc linking to sibling docs, plus those docs (#1376)."""
    pkg = tmp_path / "packages" / "coding-standards-csharp"
    pkg.mkdir(parents=True)
    (pkg / "index.md").write_text(
        "# C# Coding Standards\n\n"
        "| Topic | Doc |\n| --- | --- |\n"
        "| Repository | [C# Repository Standards](./repository.md) |\n"
        "| HTTP Client | [C# HTTP Client Standards](http-client.md) |\n"
        "| Unit Tests | [C# Unit Test Standards](unit-tests.md) |\n\n"
        "See also [external](https://example.com/x) and ![logo](./logo.png).\n"
        "Anchor: [section](./repository.md#setup).\n"
        "Wikilink: [[http-client]].\n"
    )
    (pkg / "repository.md").write_text("# C# Repository Standards\nContent.\n")
    (pkg / "http-client.md").write_text("# C# HTTP Client Standards\nContent.\n")
    (pkg / "unit-tests.md").write_text("# C# Unit Test Standards\nContent.\n")
    return pkg


def test_markdown_link_edges_emitted(tmp_path):
    """Inline/wikilink markdown links to sibling docs become references edges (#1376)."""
    pkg = _md_link_fixture(tmp_path)
    r = extract_markdown(pkg / "index.md")
    refs = [e for e in r["edges"] if e["relation"] == "references"]
    targets = {e["target"] for e in refs}
    # repository, http-client, unit-tests — each exactly once (deduped despite
    # the anchor link and wikilink pointing at repository/http-client again).
    assert len(refs) == 3, f"expected 3 reference edges, got {refs}"
    assert any("repository" in t for t in targets)
    assert any("http_client" in t for t in targets)
    assert any("unit_tests" in t for t in targets)


def test_markdown_link_skips_external_and_images(tmp_path):
    """External URLs, in-page anchors and images must not produce edges (#1376)."""
    pkg = _md_link_fixture(tmp_path)
    r = extract_markdown(pkg / "index.md")
    refs = [e for e in r["edges"] if e["relation"] == "references"]
    for e in refs:
        assert "example.com" not in e["target"]
        assert "logo" not in e["target"]


def test_markdown_link_edges_resolve_to_real_nodes(tmp_path):
    """End-to-end: after extract()'s ID remap, link targets are real doc nodes,
    so the hub doc gains edges into existing nodes instead of ghost nodes (#1376)."""
    from graphify.extract import extract
    pkg = _md_link_fixture(tmp_path)
    paths = sorted(pkg.glob("*.md"))
    res = extract(paths, cache_root=tmp_path, parallel=False)
    node_ids = {n["id"] for n in res["nodes"]}
    refs = [e for e in res["edges"] if e["relation"] == "references"]
    assert refs, "expected reference edges after full extract"
    for e in refs:
        assert e["target"] in node_ids, f"link target is a ghost node: {e}"
    # index.md must connect to all three sibling docs.
    index_id = next(n["id"] for n in res["nodes"] if n["label"] == "index.md")
    index_refs = {e["target"] for e in refs if e["source"] == index_id}
    assert len(index_refs) == 3, f"hub doc under-connected: {index_refs}"


# ── Groovy ───────────────────────────────────────────────────────────────────


def test_groovy_no_error():
    r = extract_groovy(FIXTURES / "sample.groovy")
    assert "error" not in r


def test_groovy_finds_class():
    r = extract_groovy(FIXTURES / "sample.groovy")
    assert any("SampleService" in l for l in _labels(r))


def test_groovy_finds_methods():
    r = extract_groovy(FIXTURES / "sample.groovy")
    labels = _labels(r)
    assert any("process" in l for l in labels)
    assert any("reset" in l for l in labels)


def test_groovy_finds_imports():
    r = extract_groovy(FIXTURES / "sample.groovy")
    assert "imports" in _relations(r)


def test_groovy_import_edges_have_import_context():
    r = extract_groovy(FIXTURES / "sample.groovy")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_groovy_no_dangling_edges():
    r = extract_groovy(FIXTURES / "sample.groovy")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids


def test_groovy_extends_edge():
    """`class X extends Base` must emit an inherits edge.

    tree-sitter-groovy exposes inheritance via the same `superclass` field as
    tree-sitter-java, but the inheritance handler was gated to Java only, so
    Groovy extends/implements were silently dropped.
    """
    r = extract_groovy(FIXTURES / "sample.groovy")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    found = any(
        "ExtendedService" in node_by_id.get(e["source"], "")
        and "SampleService" in node_by_id.get(e["target"], "")
        for e in r["edges"] if e["relation"] == "inherits"
    )
    assert found, "ExtendedService should have inherits edge to SampleService"


def test_groovy_implements_edge():
    """`class X implements Iface` must emit an implements edge."""
    r = extract_groovy(FIXTURES / "sample.groovy")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    found = any(
        "ExtendedService" in node_by_id.get(e["source"], "")
        and "Resettable" in node_by_id.get(e["target"], "")
        for e in r["edges"] if e["relation"] == "implements"
    )
    assert found, "ExtendedService should have implements edge to Resettable"


def test_groovy_spock_finds_class():
    r = extract_groovy(FIXTURES / "sample_spock.groovy")
    assert any("SampleSpec" in l for l in _labels(r))


def test_groovy_spock_finds_feature_methods():
    r = extract_groovy(FIXTURES / "sample_spock.groovy")
    feature_labels = [l for l in _labels(r) if l.startswith('"')]
    assert len(feature_labels) >= 2


def test_groovy_spock_finds_method_with_apostrophe():
    r = extract_groovy(FIXTURES / "sample_spock.groovy")
    assert any("it's" in l for l in _labels(r))


def test_groovy_spock_preserves_import_edges():
    r = extract_groovy(FIXTURES / "sample_spock.groovy")
    assert "imports" in _relations(r)


def test_groovy_spock_no_dangling_edges():
    r = extract_groovy(FIXTURES / "sample_spock.groovy")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids


# ── DM (BYOND DreamMaker) ────────────────────────────────────────────────────

@_needs_dm
def test_dm_no_error():
    r = extract_dm(FIXTURES / "sample.dm")
    assert "error" not in r

@_needs_dm
def test_dm_finds_global_proc():
    r = extract_dm(FIXTURES / "sample.dm")
    labels = _labels(r)
    assert any(l == "log_event()" for l in labels)
    assert any(l == "RunTest()" for l in labels)

@_needs_dm
def test_dm_finds_type_definition():
    r = extract_dm(FIXTURES / "sample.dm")
    labels = _labels(r)
    assert "/datum/weapon" in labels
    assert "/datum/weapon/sword" in labels

@_needs_dm
def test_dm_qualifies_proc_with_type_path():
    r = extract_dm(FIXTURES / "sample.dm")
    labels = _labels(r)
    assert "/datum/weapon/attack()" in labels
    assert "/datum/weapon/sword/attack()" in labels

@_needs_dm
def test_dm_finds_path_form_proc_definition():
    r = extract_dm(FIXTURES / "sample.dm")
    assert "/datum/weapon/sword/sharpen()" in _labels(r)

@_needs_dm
def test_dm_emits_include_edge():
    r = extract_dm(FIXTURES / "sample.dm")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)

@_needs_dm
def test_dm_unresolved_include_flagged_external():
    r = extract_dm(FIXTURES / "sample.dm")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    helpers = [e for e in import_edges if "helpers" in e["target"]]
    assert helpers
    assert all(e.get("external") is True for e in helpers)

@_needs_dm
def test_dm_resolves_in_file_calls():
    r = extract_dm(FIXTURES / "sample.dm")
    calls = _calls(r)
    assert any(callee == "log_event()" for _, callee in calls)
    assert ("/datum/weapon/sword/attack()", "/datum/weapon/sword/sharpen()") in calls

@_needs_dm
def test_dm_ambiguous_member_call_left_unresolved():
    r = extract_dm(FIXTURES / "sample.dm")
    calls = _calls(r)
    runtest_to_attack = [c for s, c in calls
                         if s == "RunTest()" and "attack" in c]
    assert not runtest_to_attack
    assert any(rc["callee"] == "attack" for rc in r.get("raw_calls", []))

@_needs_dm
def test_dm_emits_new_as_instantiates():
    r = extract_dm(FIXTURES / "sample.dm")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    inst = [(node_by_id.get(e["source"]), node_by_id.get(e["target"]))
            for e in r["edges"] if e["relation"] == "instantiates"]
    assert ("RunTest()", "/datum/weapon/sword") in inst

@_needs_dm
def test_dm_call_edges_have_call_context():
    r = extract_dm(FIXTURES / "sample.dm")
    call_edges = _edges_with_relation(r, "calls", "instantiates")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)

@_needs_dm
def test_dm_no_dangling_edges():
    r = extract_dm(FIXTURES / "sample.dm")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids

@_needs_dm
def test_dm_super_call_not_emitted():
    r = extract_dm(FIXTURES / "sample.dm")
    calls = _calls(r)
    assert not any(callee.strip("()") == ".." for _, callee in calls)
    assert not any(rc["callee"] == ".." for rc in r.get("raw_calls", []))


# ── DMI (BYOND icon sheets) ──────────────────────────────────────────────────

def test_dmi_no_error():
    r = extract_dmi(FIXTURES / "sample.dmi")
    assert "error" not in r

def test_dmi_emits_state_nodes():
    r = extract_dmi(FIXTURES / "sample.dmi")
    labels = _labels(r)
    assert any(l == '"mob"' for l in labels)

def test_dmi_state_contained_by_file():
    r = extract_dmi(FIXTURES / "sample.dmi")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    contains = [(node_by_id.get(e["source"]), node_by_id.get(e["target"]))
                for e in r["edges"] if e["relation"] == "contains"]
    assert ("sample.dmi", '"mob"') in contains


# ── DMM (BYOND map files) ────────────────────────────────────────────────────

def test_dmm_no_error():
    r = extract_dmm(FIXTURES / "sample.dmm")
    assert "error" not in r

def test_dmm_extracts_type_paths_as_uses_edges():
    r = extract_dmm(FIXTURES / "sample.dmm")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "uses"}
    assert "turf_closed_wall" in targets
    assert "obj_structure_table" in targets
    assert "obj_item_weapon_sword" in targets

def test_dmm_strips_var_overrides():
    r = extract_dmm(FIXTURES / "sample.dmm")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "uses"}
    assert not any("{" in t for t in targets)
    assert "obj_item_weapon_sword" in targets

def test_dmm_handles_multiline_tile_definition():
    r = extract_dmm(FIXTURES / "sample.dmm")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "uses"}
    assert "area_station_maintenance" in targets

def test_dmm_skips_grid_section():
    r = extract_dmm(FIXTURES / "sample.dmm")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "uses"}
    assert len(targets) == 5


# ── DMF (BYOND interface forms) ──────────────────────────────────────────────

def test_dmf_no_error():
    r = extract_dmf(FIXTURES / "sample.dmf")
    assert "error" not in r

def test_dmf_extracts_windows():
    r = extract_dmf(FIXTURES / "sample.dmf")
    labels = _labels(r)
    assert 'window "mapwindow"' in labels
    assert 'window "infowindow"' in labels

def test_dmf_elem_labels_carry_control_type():
    r = extract_dmf(FIXTURES / "sample.dmf")
    labels = _labels(r)
    assert 'elem "map" [MAP]' in labels

def test_dmf_elem_under_window():
    r = extract_dmf(FIXTURES / "sample.dmf")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    contains = [(node_by_id.get(e["source"]), node_by_id.get(e["target"]))
                for e in r["edges"] if e["relation"] == "contains"]
    assert ('window "mapwindow"', 'elem "map" [MAP]') in contains

def test_dmf_no_dangling_edges():
    r = extract_dmf(FIXTURES / "sample.dmf")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids
        assert e["target"] in node_ids


# -- .NET project files (.sln, .csproj, .xaml, .razor) ------------------------

def test_sln_no_error():
    r = extract_sln(FIXTURES / "sample.sln")
    assert "error" not in r

def test_sln_finds_projects():
    r = extract_sln(FIXTURES / "sample.sln")
    labels = _labels(r)
    assert any("WebApi" in l for l in labels)
    assert any("Domain" in l for l in labels)

def test_sln_contains_edges():
    r = extract_sln(FIXTURES / "sample.sln")
    assert "contains" in _relations(r)

def test_sln_project_dependency_edges():
    r = extract_sln(FIXTURES / "sample.sln")
    assert "imports" in _relations(r)

def test_csproj_no_error():
    r = extract_csproj(FIXTURES / "sample.csproj")
    assert "error" not in r

def test_csproj_finds_packages():
    r = extract_csproj(FIXTURES / "sample.csproj")
    labels = _labels(r)
    assert any("MediatR" in l for l in labels)
    assert any("FluentValidation" in l for l in labels)

def test_csproj_finds_project_references():
    r = extract_csproj(FIXTURES / "sample.csproj")
    labels = _labels(r)
    assert any("Domain.csproj" in l for l in labels)

def test_csproj_finds_target_framework():
    r = extract_csproj(FIXTURES / "sample.csproj")
    assert any("net8.0" in l for l in _labels(r))

def test_csproj_finds_sdk():
    r = extract_csproj(FIXTURES / "sample.csproj")
    assert any("Microsoft.NET.Sdk.Web" in l for l in _labels(r))

def test_xaml_finds_class_and_event_references():
    r = extract_xaml(FIXTURES / "sample.xaml")
    assert "error" not in r
    assert "MainWindow" in _labels(r)
    assert any(e["relation"] == "references" and e.get("context") == "event" for e in r["edges"])

def test_razor_no_error():
    r = extract_razor(FIXTURES / "sample.razor")
    assert "error" not in r

def test_razor_finds_using_directives():
    r = extract_razor(FIXTURES / "sample.razor")
    assert "imports" in _relations(r)

def test_razor_finds_component_references():
    r = extract_razor(FIXTURES / "sample.razor")
    assert "calls" in _relations(r)

def test_razor_finds_inherits():
    r = extract_razor(FIXTURES / "sample.razor")
    assert "inherits" in _relations(r)

def test_razor_finds_code_block_methods():
    r = extract_razor(FIXTURES / "sample.razor")
    labels = _labels(r)
    assert any("IncrementCount" in l for l in labels)
    assert any("LoadData" in l for l in labels)

def test_razor_no_dangling_edges():
    r = extract_razor(FIXTURES / "sample.razor")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids


# ---------------Salesforce Apex (.cls / .trigger)----------------------

def test_apex_class_extraction():
    r = extract_apex(FIXTURES / "sample.cls")
    labels = _labels(r)
    assert "AccountService" in labels

def test_apex_enum_extraction():
    r = extract_apex(FIXTURES / "sample.cls")
    labels = _labels(r)
    assert "AccountStatus" in labels

def test_apex_interface_extraction():
    r = extract_apex(FIXTURES / "sample.cls")
    labels = _labels(r)
    assert "Notifiable" in labels

def test_apex_interface_extends(tmp_path):
    source = tmp_path / "PaymentProcessor.cls"
    source.write_text(
        "public interface PaymentProcessor extends Processor, Auditable { void process(); }\n"
    )
    result = extract_apex(source)
    inheritance = _edge_labels(result, "extends") | _edge_labels(result, "implements")
    assert ("PaymentProcessor", "Processor") in inheritance
    assert ("PaymentProcessor", "Auditable") in inheritance

def test_apex_method_extraction():
    r = extract_apex(FIXTURES / "sample.cls")
    labels = _labels(r)
    assert any("getAccounts" in l for l in labels)
    assert any("updateAccountsAsync" in l for l in labels)
    assert any("createAccounts" in l for l in labels)
    assert any("deleteOldAccounts" in l for l in labels)

def test_apex_contains_and_method_relations():
    r = extract_apex(FIXTURES / "sample.cls")
    relations = _relations(r)
    assert "contains" in relations
    assert "method" in relations

def test_apex_soql_uses_edge():
    r = extract_apex(FIXTURES / "sample.cls")
    relations = _relations(r)
    assert "uses" in relations
    labels = _labels(r)
    assert "Account" in labels

def test_apex_dml_uses_edge():
    r = extract_apex(FIXTURES / "sample.cls")
    dml_labels = {n["label"] for n in r["nodes"] if n["label"] in ("insert", "update", "delete", "upsert")}
    assert len(dml_labels) > 0

def test_apex_file_node_present():
    r = extract_apex(FIXTURES / "sample.cls")
    labels = _labels(r)
    assert "sample.cls" in labels

def test_apex_trigger_extraction():
    r = extract_apex(FIXTURES / "sample.trigger")
    labels = _labels(r)
    assert "sample.trigger" in labels
    assert "AccountTrigger" in labels

def test_apex_trigger_uses_sobject():
    r = extract_apex(FIXTURES / "sample.trigger")
    relations = _relations(r)
    assert "uses" in relations
    labels = _labels(r)
    assert "Account" in labels

def test_apex_missing_file_returns_empty():
    r = extract_apex(Path("nonexistent.cls"))
    assert r["nodes"] == []
    assert r["edges"] == []

def test_apex_no_dangling_edges():
    for fixture in ("sample.cls", "sample.trigger"):
        r = extract_apex(FIXTURES / fixture)
        node_ids = {n["id"] for n in r["nodes"]}
        for e in r["edges"]:
            assert e["source"] in node_ids, f"dangling source in {fixture}: {e}"
            assert e["target"] in node_ids, f"dangling target in {fixture}: {e}"


# -- SystemVerilog -------------------------------------------------------------

def test_systemverilog_no_error():
    r = extract_verilog(FIXTURES / "sample.sv")
    assert "error" not in r


def test_systemverilog_splits_inherits_and_implements():
    r = extract_verilog(FIXTURES / "sample.sv")
    assert ("DataProcessor", "BaseProcessor") in _edge_labels(r, "inherits")
    assert ("DataProcessor", "Processor") in _edge_labels(r, "implements")


def test_systemverilog_field_parameter_return_and_generic_contexts():
    r = extract_verilog(FIXTURES / "sample.sv")
    assert ("DataProcessor", "Result") in _edge_labels(r, "references", "field")
    assert ("DataProcessor", "Payload") in _edge_labels(r, "references", "generic_arg")
    assert ("build", "Payload") in _edge_labels(r, "references", "parameter_type")
    assert ("build", "Result") in _edge_labels(r, "references", "return_type")
    assert ("build", "Payload") in _edge_labels(r, "references", "generic_arg")


def test_systemverilog_qualified_field_references():
    """Class properties with leading qualifiers (rand/local/protected/etc.) must
    still emit `references` field edges. The field regex only matched unqualified
    `<type> <name>;` declarations, so `rand Config x;` (three tokens) failed to
    match and its type reference was silently dropped.
    """
    r = extract_verilog(FIXTURES / "sample.sv")
    field_refs = _edge_labels(r, "references", "field")
    assert ("DataProcessor", "Config") in field_refs, "rand-qualified field dropped"
    assert ("DataProcessor", "BaseProcessor") in field_refs, "protected-qualified field dropped"


def test_systemverilog_does_not_emit_type_parameter_refs():
    r = extract_verilog(FIXTURES / "sample.sv")
    assert ("Result", "T") not in _edge_labels(r, "references", "field")


def test_systemverilog_preserves_existing_module_extraction():
    r = extract_verilog(FIXTURES / "sample.sv")
    labels = set(_labels(r))
    assert {"top", "leaf", "add()", "tick"}.issubset(labels)
    assert "imports_from" in _relations(r)
    assert "instantiates" in _relations(r)


def test_systemverilog_missing_file_returns_empty():
    r = extract_verilog(Path("nonexistent.sv"))
    assert r["nodes"] == []
    assert r["edges"] == []


def test_systemverilog_no_dangling_edges():
    r = extract_verilog(FIXTURES / "sample.sv")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"dangling source: {e}"
        assert e["target"] in node_ids, f"dangling target: {e}"


# ── BSL (1C / OneScript) ─────────────────────────────────────────────────────

def test_bsl_no_error():
    r = extract_bsl(FIXTURES / "sample.bsl")
    assert "error" not in r


def test_bsl_finds_procedures_and_functions():
    r = extract_bsl(FIXTURES / "sample.bsl")
    labels = _labels(r)
    assert "ОбработатьЗаказ()" in labels
    assert "ВычислитьСумму()" in labels


def test_bsl_extracts_call_graph():
    # ОбработатьЗаказ() calls ВычислитьСумму() by bare name in the same module.
    r = extract_bsl(FIXTURES / "sample.bsl")
    assert ("ОбработатьЗаказ()", "ВычислитьСумму()") in _calls(r)


def test_bsl_new_expression_references_platform_type():
    r = extract_bsl(FIXTURES / "sample.bsl")
    # _edge_labels normalizes labels (strips the "()" suffix).
    refs = _edge_labels(r, "references", "new")
    assert ("ОбработатьЗаказ", "ТаблицаЗначений") in refs
    assert ("ОбработатьЗаказ", "Запрос") in refs


def test_bsl_member_call_not_resolved_cross_module():
    # Заказ.Записать() is a member call → recorded as is_member_call so it is
    # not falsely resolved to a free procedure of the same name elsewhere.
    r = extract_bsl(FIXTURES / "sample.bsl")
    member = [rc for rc in r["raw_calls"] if rc["callee"] == "Записать"]
    assert member and all(rc["is_member_call"] for rc in member)


def test_bsl_no_dangling_edges():
    r = extract_bsl(FIXTURES / "sample.bsl")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"dangling source: {e}"
        if e["relation"] != "imports":
            assert e["target"] in node_ids, f"dangling target: {e}"


def test_bsl_onescript_use_directive_imports():
    r = extract_bsl(FIXTURES / "sample.os")
    assert "imports" in _relations(r)
    # Cross-procedure call inside a OneScript module still resolves.
    assert ("ГлавныйМетод()", "Запустить()") in _calls(r)


# ── 1C:EDT metadata (.mdo) ───────────────────────────────────────────────────

EDT = FIXTURES / "edt"
CATALOG_MDO = EDT / "Catalogs" / "Контрагенты" / "Контрагенты.mdo"
ENUM_MDO = EDT / "Enums" / "СтатусДоговора" / "СтатусДоговора.mdo"
CONFIG_MDO = EDT / "Configuration" / "Configuration.mdo"
COMMON_COMMAND_MDO = EDT / "CommonCommands" / "ОбщаяКоманда" / "ОбщаяКоманда.mdo"
DOCUMENT_MDO = EDT / "Documents" / "Заказ" / "Заказ.mdo"
FUNCOPT_MDO = EDT / "FunctionalOptions" / "УчетСкидок" / "УчетСкидок.mdo"
EXCHANGE_MDO = EDT / "ExchangePlans" / "Обмен" / "Обмен.mdo"
SUBSCRIPTION_MDO = (EDT / "EventSubscriptions" / "ПриЗаписиКонтрагента"
                    / "ПриЗаписиКонтрагента.mdo")
PARTIAL_SUBSCRIPTION_MDO = (EDT / "EventSubscriptions" / "БезОбработчика"
                            / "БезОбработчика.mdo")
HTTP_SERVICE_MDO = EDT / "HTTPServices" / "Обмен" / "Обмен.mdo"
WEB_SERVICE_MDO = EDT / "WebServices" / "Каталог" / "Каталог.mdo"
JOURNAL_MDO = EDT / "DocumentJournals" / "ЖурналПродаж" / "ЖурналПродаж.mdo"
REPORT_FORM = EDT / "Reports" / "ВзаиморасчетыОтчет" / "Forms" / "ФормаОтчета" / "Form.form"
JOURNAL_FORM = (EDT / "DocumentJournals" / "ЖурналПродаж" / "Forms"
                / "ФормаСписка" / "Form.form")
CALCREG_MDO = EDT / "CalculationRegisters" / "РегистрРасчета1" / "РегистрРасчета1.mdo"
RECALC_MDO = (EDT / "CalculationRegisters" / "РегистрРасчета1" / "Recalculations"
              / "Перерасчет" / "Перерасчет.mdo")
EDS_TABLE_MDO = (EDT / "ExternalDataSources" / "ТекущаяСУБД" / "Tables"
                 / "ИнфоТаблица" / "ИнфоТаблица.mdo")
EDS_CUBE_MDO = EDT / "ExternalDataSources" / "ТекущаяСУБД" / "Cubes" / "Куб1" / "Куб1.mdo"
EDS_DIMTABLE_MDO = (EDT / "ExternalDataSources" / "ТекущаяСУБД" / "Cubes" / "Куб1"
                    / "DimensionTables" / "Измерение1" / "Измерение1.mdo")


def test_edt_mdo_no_error():
    assert "error" not in extract_edt_mdo(CATALOG_MDO)


def test_edt_mdo_object_node():
    assert "Catalog.Контрагенты" in _labels(extract_edt_mdo(CATALOG_MDO))


def test_edt_mdo_contains_children():
    r = extract_edt_mdo(CATALOG_MDO)
    labels = _labels(r)
    assert "ИНН" in labels             # attribute
    assert "Контакты" in labels        # tabular section
    assert "ФормаЭлемента" in labels   # form
    assert "Печать" in labels          # command
    contains = _edge_labels(r, "contains")
    assert ("Catalog.Контрагенты", "ИНН") in contains
    assert ("Catalog.Контрагенты", "ФормаЭлемента") in contains


def test_edt_mdo_defines_sibling_modules():
    r = extract_edt_mdo(CATALOG_MDO)
    obj_id = _node_by_label(r, "Catalog.Контрагенты")["id"]
    obj_defines = {e["target"] for e in r["edges"]
                   if e["relation"] == "defines" and e["source"] == obj_id}
    # ObjectModule.bsl and ManagerModule.bsl both exist next to the .mdo.
    assert len(obj_defines) >= 2


def test_edt_mdo_form_defines_its_module():
    r = extract_edt_mdo(CATALOG_MDO)
    form_id = _node_by_label(r, "ФормаЭлемента")["id"]
    assert any(e["relation"] == "defines" and e["source"] == form_id
               for e in r["edges"])


def test_edt_mdo_captures_uuid():
    # Object and child uuids land as node attributes (id stays name-based).
    r = extract_edt_mdo(CATALOG_MDO)
    obj = _node_by_label(r, "Catalog.Контрагенты")
    assert obj["uuid"] == "11111111-1111-4111-8111-111111111111"
    attr = _node_by_label(r, "ИНН")
    assert attr["uuid"] == "66666666-6666-4666-8666-666666666666"
    # The id is unchanged by uuid capture, so code↔metadata merge still holds.
    assert "uuid" not in obj["id"]


def test_edt_configuration_captures_uuid():
    r = extract_edt_mdo(CONFIG_MDO)
    conf = _node_by_label(r, "ТестоваяКонфигурация")
    assert conf.get("uuid")


def test_edt_mdo_enum_values():
    r = extract_edt_mdo(ENUM_MDO)
    labels = _labels(r)
    assert "Enum.СтатусДоговора" in labels
    assert "Действует" in labels
    assert "Закрыт" in labels


def test_edt_configuration_registers_children():
    r = extract_edt_mdo(CONFIG_MDO)
    labels = _labels(r)
    assert "ТестоваяКонфигурация" in labels          # Configuration node label = <name>
    contains = _edge_labels(r, "contains")
    assert ("ТестоваяКонфигурация", "Catalog.Контрагенты") in contains
    assert ("ТестоваяКонфигурация", "Enum.СтатусДоговора") in contains
    # Non-FQN scalar children (scriptVariant, compatibilityMode) create no nodes.
    assert "Russian" not in labels


def test_edt_configuration_registers_newly_recognised_kinds():
    """Kinds the skill documents that the parser used to drop silently.

    Asserted on the `contains` edge rather than on node presence: an object .mdo
    becomes a node from its root tag regardless of _EDT_KIND_PREFIXES, so a
    dropped registration surfaces as an orphan, not as a missing node — a
    node-presence assertion would pass against the unfixed parser.
    """
    contains = _edge_labels(extract_edt_mdo(CONFIG_MDO), "contains")
    for fqn in ("DocumentJournal.ЖурналПродаж",
                "SettingsStorage.ХранилищеОтчетов",
                "PaletteColor.ФирменныйСиний",
                "ExternalDataSource.ТекущаяСУБД"):
        assert ("ТестоваяКонфигурация", fqn) in contains


def test_edt_configuration_ignores_non_fqn_children():
    labels = _labels(extract_edt_mdo(CONFIG_MDO))
    assert "8.3.24" not in labels     # <compatibilityMode>
    assert "Russian" not in labels    # <scriptVariant>


def test_edt_nested_unknown_root_tag_keeps_flat_id(tmp_path):
    """An unrecognised layout must not get an invented FQN.

    The recursive resolver keys off folder markers, but the root tag of the .mdo
    is what names the SubKind. When the two disagree the object is something the
    marker map does not model, so the parser falls back to the flat id instead of
    asserting a parentage it cannot justify.
    """
    d = tmp_path / "ExternalDataSources" / "СУБД" / "Tables" / "Стол"
    d.mkdir(parents=True)
    mdo = d / "Стол.mdo"
    mdo.write_text(
        '''<?xml version="1.0" encoding="UTF-8"?>
<mdclass:Sequence xmlns:mdclass="http://g5.1c.ru/v8/dt/metadata/mdclass"
                  uuid="dddddddd-0000-4000-8000-000000000001">
  <name>Стол</name>
</mdclass:Sequence>
''',
        encoding="utf-8",
    )
    labels = _labels(extract_edt_mdo(mdo))
    assert "Sequence.Стол" in labels
    assert not [l for l in labels if l.startswith("ExternalDataSource.")]


def test_edt_recalculation_fqn_carries_its_register():
    """A Recalculation is a child of its CalculationRegister, not a top object.

    The root tag of the standalone .mdo is `Recalculation`, so a flat id would
    read `Recalculation.Перерасчет` — a name two registers can share.
    """
    r = extract_edt_mdo(RECALC_MDO)
    assert "CalculationRegister.РегистрРасчета1.Recalculation.Перерасчет" in _labels(r)
    assert ("CalculationRegister.РегистрРасчета1",
            "CalculationRegister.РегистрРасчета1.Recalculation.Перерасчет")         in _edge_labels(r, "contains")


def test_edt_recalculation_registered_inline_by_parent():
    """The parent register declares its recalculations inline (skill §4.5)."""
    contains = _edge_labels(extract_edt_mdo(CALCREG_MDO), "contains")
    assert ("CalculationRegister.РегистрРасчета1", "Перерасчет") in contains


def test_edt_external_data_source_children_fqn():
    """Tables and cubes nest under the data source; there is no `Table` kind in 1C."""
    t = extract_edt_mdo(EDS_TABLE_MDO)
    assert "ExternalDataSource.ТекущаяСУБД.Table.ИнфоТаблица" in _labels(t)
    assert ("ExternalDataSource.ТекущаяСУБД",
            "ExternalDataSource.ТекущаяСУБД.Table.ИнфоТаблица") in _edge_labels(t, "contains")

    c = extract_edt_mdo(EDS_CUBE_MDO)
    assert "ExternalDataSource.ТекущаяСУБД.Cube.Куб1" in _labels(c)

    # A dimension table nests one level deeper still — the grammar is recursive.
    d = extract_edt_mdo(EDS_DIMTABLE_MDO)
    assert "ExternalDataSource.ТекущаяСУБД.Cube.Куб1.DimensionTable.Измерение1" in _labels(d)


def test_edt_external_table_fields_are_field_not_attribute():
    """Fields of an external table carry SubKind `Field` (skill §4.21).

    The two owners also spell the container differently: <tableFields> on a
    Table, <fields> on a DimensionTable.
    """
    t = extract_edt_mdo(EDS_TABLE_MDO)
    assert _make_id("ExternalDataSource", "ТекущаяСУБД", "Table", "ИнфоТаблица",
                    "Field", "Поле1") in {n["id"] for n in t["nodes"]}
    assert _make_id("ExternalDataSource", "ТекущаяСУБД", "Table", "ИнфоТаблица",
                    "Attribute", "Поле1") not in {n["id"] for n in t["nodes"]}

    d = extract_edt_mdo(EDS_DIMTABLE_MDO)
    assert _make_id("ExternalDataSource", "ТекущаяСУБД", "Cube", "Куб1",
                    "DimensionTable", "Измерение1", "Field",
                    "ПолеИзмерения") in {n["id"] for n in d["nodes"]}


def test_edt_existing_ids_unchanged_by_nesting_support():
    """Regression: previously supported objects keep their exact ids.

    Literal values, not recomputed ones — the point is that the id a stored graph
    already holds still resolves, so recomputing through _make_id would assert
    nothing.
    """
    assert _node_by_label(extract_edt_mdo(CATALOG_MDO),
                          "Catalog.Контрагенты")["id"] == "catalog_контрагенты"
    assert _node_by_label(extract_edt_mdo(ENUM_MDO),
                          "Enum.СтатусДоговора")["id"] == "enum_статусдоговора"
    assert _node_by_label(extract_edt_mdo(SUBSYSTEM_MDO),
                          "Subsystem.Продажи")["id"] == "subsystem_продажи"
    assert _node_by_label(extract_edt_mdo(NESTED_SUBSYSTEM_MDO),
                          "Subsystem.Продажи.Розница")["id"] == "subsystem_продажи_розница"


def test_edt_common_command_binds_its_own_module():
    """A CommonCommand keeps CommandModule.bsl in its own folder (skill §4.15).

    Every other kind puts its command module under Commands/<C>/, which is why
    this one was the only binding missing.
    """
    r = extract_edt_mdo(COMMON_COMMAND_MDO)
    defines = [e for e in r["edges"] if e["relation"] == "defines"]
    assert len(defines) == 1
    assert defines[0]["source"] == _node_by_label(r, "CommonCommand.ОбщаяКоманда")["id"]


def test_edt_object_command_module_is_bound_once_and_from_the_command():
    """An object's command module hangs off the command, not off the object.

    This is the guard on the fix above: putting CommandModule.bsl into
    _EDT_OBJECT_MODULES would hand every object a second edge to a file that
    belongs to its command.
    """
    r = extract_edt_mdo(CATALOG_MDO)
    module = CATALOG_MDO.parent / "Commands" / "Печать" / "CommandModule.bsl"
    bound = [e for e in r["edges"]
             if e["relation"] == "defines" and e["target"] == _make_id(str(module))]
    assert len(bound) == 1
    assert bound[0]["source"] == _make_id("Catalog", "Контрагенты", "Command", "Печать")


def test_edt_configuration_binds_its_own_modules():
    """Configuration modules live beside Configuration.mdo, not beside an object.

    The Configuration branch returns before the shared module loop, so none of
    the five (skill §1) was ever linked; the fixture carries two of them.
    """
    r = extract_edt_mdo(CONFIG_MDO)
    defines = [e for e in r["edges"] if e["relation"] == "defines"]
    conf_id = _node_by_label(r, "ТестоваяКонфигурация")["id"]
    assert len(defines) == 2
    assert {e["source"] for e in defines} == {conf_id}
    assert {_make_id(str(CONFIG_MDO.parent / n)) for n in
            ("ManagedApplicationModule.bsl", "SessionModule.bsl")} ==         {e["target"] for e in defines}


def test_edt_configuration_contains_edges_unchanged_by_module_binding():
    """Regression: registrations are untouched by the module fix.

    Literal expected set — recomputing it from the same code would assert
    nothing about whether the registrations still resolve.
    """
    r = extract_edt_mdo(CONFIG_MDO)
    labels = {n["id"]: n["label"] for n in r["nodes"]}
    contained = sorted(labels[e["target"]] for e in r["edges"]
                       if e["relation"] == "contains")
    assert contained == [
        "CalculationRegister.РегистрРасчета1",
        "Catalog.Контрагенты",
        "Catalog.Пользователи",
        "CommonCommand.ОбщаяКоманда",
        "Document.Заказ",
        "DocumentJournal.ЖурналПродаж",
        "Enum.СтатусДоговора",
        "EventSubscription.ПриЗаписиКонтрагента",
        "ExchangePlan.Обмен",
        "ExternalDataSource.ТекущаяСУБД",
        "FunctionalOption.УчетСкидок",
        "HTTPService.Обмен",
        "PaletteColor.ФирменныйСиний",
        "Report.ВзаиморасчетыОтчет",
        "Role.Менеджер",
        "Role.ПолныеПрава",
        "SettingsStorage.ХранилищеОтчетов",
        "Subsystem.Продажи",
        "WebService.Каталог",
    ]


def test_edt_form_owner_resolves_for_every_kind_that_owns_a_folder():
    """A form under a folder the map did not know lost its owner entirely.

    _EDT_PLURAL_TO_KIND covered 17 of the 48 kinds, so extract_edt_form returned
    an empty result for anything else — no node, no edges, no diagnostic.
    """
    r = extract_edt_form(JOURNAL_FORM)
    assert r["nodes"], "форма без владельца не даёт вообще ничего"
    assert _node_by_label(r, "ФормаСписка")["id"] == _make_id(
        "DocumentJournal", "ЖурналПродаж", "Form", "ФормаСписка")
    # …and it merges with the id the owner .mdo emits for the same form.
    owner = extract_edt_mdo(EDT / "DocumentJournals" / "ЖурналПродаж" / "ЖурналПродаж.mdo")
    assert _node_by_label(owner, "ФормаСписка")["id"] == _node_by_label(r, "ФормаСписка")["id"]


def test_edt_form_under_an_unknown_folder_invents_nothing(tmp_path):
    """An unmapped folder must yield no owner rather than a guessed one."""
    d = tmp_path / "src" / "ВыдуманныйВид" / "Ы" / "Forms" / "Ф"
    d.mkdir(parents=True)
    form = d / "Form.form"
    form.write_text(
        '''<?xml version="1.0" encoding="UTF-8"?>
<form:Form xmlns:form="http://g5.1c.ru/v8/dt/form"/>
''',
        encoding="utf-8",
    )
    r = extract_edt_form(form)
    assert r == {"nodes": [], "edges": []}


def test_edt_type_flavours_all_resolve_to_one_object():
    """A type flavour says in which capacity an object is used, not which object.

    All eight flavours of `Справочник.Клиенты` denote the same catalog; giving
    each its own node would split one object into seven.
    """
    for prefix in ("CatalogRef", "CatalogObject", "CatalogManager", "CatalogList",
                   "CatalogSelection"):
        assert _edt_type_kind(prefix) == "Catalog"
    for prefix in ("InformationRegisterRecordSet", "InformationRegisterRecordKey",
                   "InformationRegisterRecordManager"):
        assert _edt_type_kind(prefix) == "InformationRegister"


def test_edt_type_flavour_stripping_takes_the_longest_suffix():
    """The greedy trap: two flavours can both match one prefix.

    `InformationRegisterRecordManager` ends in `RecordManager` and in `Manager`;
    stripping the shorter one leaves `InformationRegisterRecord`, which is not a
    kind. `ChartOfCharacteristicTypesObject` is the mirror case — the kind name
    itself ends in a word that looks like part of a flavour.
    """
    assert _edt_type_kind("InformationRegisterRecordManager") == "InformationRegister"
    assert _edt_type_kind("ChartOfCharacteristicTypesObject") == "ChartOfCharacteristicTypes"


def test_edt_type_remainder_that_is_not_a_kind_is_not_a_reference():
    """Anything ending in a flavour word is not thereby a metadata reference."""
    for prefix in ("String", "DynamicList", "ValueList", "Listendruck", "ValueTable"):
        assert _edt_type_kind(prefix) is None


def test_edt_form_links_its_main_attribute_to_the_owning_object():
    """The main form attribute is typed `<Kind>Object.<Name>` (skill §5, rule 6).

    It matched nothing while only `*Ref.` was parsed, which is why 719 typed
    references were missing on the corpus. Asserted on a report form, where the
    Object flavour is the only source of the target: the catalog form also
    names its object in <mainTable>, so it would pass either way.
    """
    r = extract_edt_form(REPORT_FORM)
    assert "Report.ВзаиморасчетыОтчет" in {n["label"] for n in r["nodes"]}


def test_edt_tabular_section_attribute_is_reachable():
    """A tabular-section column hangs off the section, not off the object.

    The child loop was flat, so `Kind.Name.TabularSection.T.Attribute.A` — a
    plain application of the recursive FQN grammar (skill §3) — did not exist;
    on the corpus that is 1131 columns in 176 sections.
    """
    r = extract_edt_mdo(CATALOG_MDO)
    column = _make_id("Catalog", "Контрагенты", "TabularSection", "Контакты",
                      "Attribute", "Значение")
    assert column in {n["id"] for n in r["nodes"]}
    section = _make_id("Catalog", "Контрагенты", "TabularSection", "Контакты")
    assert [e["source"] for e in r["edges"] if e["target"] == column] == [section]


def test_edt_non_container_children_do_not_descend():
    """Only documented containers descend; other blocks own nothing.

    An attribute carries <type><types>String</types></type>; recursing
    unconditionally would mint a node for the type and for every other nested
    block that names nothing.
    """
    r = extract_edt_mdo(CATALOG_MDO)
    attribute = _make_id("Catalog", "Контрагенты", "Attribute", "ИНН")
    assert attribute in {n["id"] for n in r["nodes"]}
    assert [e for e in r["edges"] if e["source"] == attribute] == []
    assert "String" not in {n["label"] for n in r["nodes"]}


def _refs(result):
    """(source label, target label, context) for every `references` edge."""
    labels = {n["id"]: n["label"] for n in result["nodes"]}
    return {(labels.get(e["source"]), labels.get(e["target"]), e.get("context"))
            for e in result["edges"] if e["relation"] == "references"}


def test_edt_reference_tag_map_covers_the_documented_tags():
    """The map IS the whitelist; a gap in it is a silently dropped class."""
    covered = set(_EDT_REF_TAGS) | set(_EDT_REF_CONTAINER_TAGS) | {"references", "columns"}
    for tag in ("registerRecords", "basedOn", "owners", "sequences", "inputByString",
                "defaultObjectForm", "defaultListForm", "defaultChoiceForm",
                "characteristicExtValues", "chartOfAccounts", "registeredDocuments",
                "references", "mainDataCompositionSchema", "location", "content",
                "handler", "source", "methodName", "task", "addressing",
                "mainAddressingAttribute", "addressingDimension",
                "commandParameterType", "defaultRoles", "defaultLanguage"):
        assert tag in covered, tag


def test_edt_declared_reference_becomes_an_edge():
    refs = _refs(extract_edt_mdo(DOCUMENT_MDO))
    assert ("Document.Заказ", "AccumulationRegister.ТоварыНаСкладах",
            "register-records") in refs
    assert ("Document.Заказ", "Catalog.Контрагенты", "based-on") in refs


def test_edt_unknown_tag_and_unparseable_value_yield_nothing():
    """Two ways to invent an edge, both refused.

    The fixture carries a tag outside the map whose value looks exactly like an
    FQN, and a <version> that would parse as kind "3" under any shape-only
    heuristic.
    """
    refs = _refs(extract_edt_mdo(DOCUMENT_MDO))
    assert not [r for r in refs if r[1] and r[1].startswith("3.")]
    assert len([r for r in refs if r[1] == "Catalog.Контрагенты"]) == 1


def test_edt_membership_content_is_references_not_contains():
    """A functional option does not OWN the document whose visibility it drives.

    Same for an exchange plan and the catalogs it replicates. `contains` there
    would compete with the real ownership edge from Configuration.
    """
    fo = extract_edt_mdo(FUNCOPT_MDO)
    assert ("FunctionalOption.УчетСкидок", "Document.Заказ",
            "functional-option-content") in _refs(fo)
    assert [e for e in fo["edges"] if e["relation"] == "contains"] == []

    # ExchangePlan spells the SAME tag as a container: <content><mdObject>...
    ep = extract_edt_mdo(EXCHANGE_MDO)
    assert ("ExchangePlan.Обмен", "Catalog.Контрагенты",
            "exchange-plan-content") in _refs(ep)


def test_edt_subsystem_content_still_contains_and_unchanged():
    """Regression: the one <content> that IS ownership keeps its relation."""
    r = extract_edt_mdo(SUBSYSTEM_MDO)
    labels = {n["id"]: n["label"] for n in r["nodes"]}
    contained = sorted((labels[e["source"]], labels[e["target"]])
                       for e in r["edges"] if e["relation"] == "contains")
    assert contained == [
        ("Subsystem.Продажи", "Catalog.Контрагенты"),
        ("Subsystem.Продажи", "Enum.СтатусДоговора"),
        ("Subsystem.Продажи", "Subsystem.Продажи.Розница"),
    ]
    assert [e for e in r["edges"] if e["relation"] == "references"] == []


def test_edt_value_forms_are_parsed_separately():
    """One string shape, four meanings, decided by the tag it came from."""
    assert _edt_ref_target("object", "Document.Заказ") == (["Document", "Заказ"], None)
    assert _edt_ref_target("member", "Catalog.X.StandardAttribute.Description") == (
        ["Catalog", "X", "StandardAttribute", "Description"], None)
    assert _edt_ref_target("method", "CommonModule.Обмен.Выполнить") == (
        ["CommonModule", "Обмен"], "Выполнить")
    assert _edt_ref_target("type", "CatalogObject.Клиенты") == (["Catalog", "Клиенты"], None)
    assert _edt_ref_target("object", "3.2.7.38") is None


def test_edt_input_by_string_targets_the_member_not_the_object():
    assert ("Document.Заказ", "Document.Заказ.StandardAttribute.Number",
            "input-by-string") in _refs(extract_edt_mdo(DOCUMENT_MDO))


def test_edt_reference_to_a_member_nobody_extracts_yet_is_stubbed():
    """A reference is not dropped because its target has no extractor yet.

    The stub carries the id the real node will get, so it merges rather than
    duplicating once that extraction lands.
    """
    stub = _make_id("Document", "Заказ", "StandardAttribute", "Number")
    ids = [n["id"] for n in extract_edt_mdo(DOCUMENT_MDO)["nodes"]]
    assert stub in ids and ids.count(stub) == 1


def test_edt_attribute_type_reference_belongs_to_the_attribute():
    """Precision over aggregate: the aggregate follows, the reverse does not."""
    r = extract_edt_mdo(CATALOG_MDO)
    attribute = _make_id("Catalog", "Контрагенты", "Attribute", "ОсновнойМенеджер")
    target = _make_id("Catalog", "Пользователи")
    typed = [e for e in r["edges"]
             if e["relation"] == "references" and e["target"] == target]
    assert [e["source"] for e in typed] == [attribute]


def test_edt_journal_column_reference_belongs_to_the_column():
    r = extract_edt_mdo(JOURNAL_MDO)
    column = _make_id("DocumentJournal", "ЖурналПродаж", "Column", "Организация")
    assert ("Организация", "Document.Заказ.Attribute.Организация",
            "journal-column-source") in _refs(r)
    assert column in {n["id"] for n in r["nodes"]}


def test_edt_event_subscription_is_read_as_a_triple():
    """Alone each of the three tags says little; the meaning is in the triple."""
    refs = _refs(extract_edt_mdo(SUBSCRIPTION_MDO))
    assert ("EventSubscription.ПриЗаписиКонтрагента", "Catalog.Контрагенты",
            "event-source") in refs
    handler = [r for r in refs if r[1] == "CommonModule.ОбработчикиСобытий"]
    assert len(handler) == 1
    assert handler[0][2] == "event-handler:OnWrite:ПриЗаписи"


def test_edt_incomplete_subscription_yields_what_is_determinable():
    refs = _refs(extract_edt_mdo(PARTIAL_SUBSCRIPTION_MDO))
    assert ("EventSubscription.БезОбработчика", "Catalog.Контрагенты",
            "event-source") in refs
    assert not [r for r in refs if r[1] and r[1].startswith("CommonModule.")]


def test_edt_service_handler_resolves_a_bare_procedure_name():
    """A service names its handler by bare procedure name, not by FQN.

    The skill spells out the contrast (§4.22): an EventSubscription's <handler>
    is a full method FQN in someone else's module, while a service's is a bare
    name implemented in the Module.bsl beside the .mdo. Parsing only the FQN
    form covered 8 handlers out of 98 on a real configuration.
    """
    mdo = extract_edt_mdo(HTTP_SERVICE_MDO)
    handler = [e for e in mdo["edges"] if e.get("context") == "service-handler"]
    assert len(handler) == 1
    assert handler[0]["source"] == _make_id(
        "HTTPService", "Обмен", "URLTemplate", "ШаблонЗаказа", "Method", "POST")
    # The target is the real procedure node, not a dangling id: extract_bsl
    # gives that procedure the same id, so the two halves merge.
    module = extract_bsl(HTTP_SERVICE_MDO.parent / "Module.bsl")
    assert handler[0]["target"] in {n["id"] for n in module["nodes"]}


def test_edt_web_service_operation_resolves_its_procedure():
    r = extract_edt_mdo(WEB_SERVICE_MDO)
    proc = [e for e in r["edges"] if e.get("context") == "service-procedure"]
    assert len(proc) == 1
    assert proc[0]["source"] == _make_id(
        "WebService", "Каталог", "Operation", "ПолучитьНоменклатуру")
    module = extract_bsl(WEB_SERVICE_MDO.parent / "Module.bsl")
    assert proc[0]["target"] in {n["id"] for n in module["nodes"]}


def test_edt_event_subscription_handler_is_not_treated_as_a_bare_name():
    """The two <handler> forms must not be confused for one another.

    A dotted value is a method FQN in another module and must never be resolved
    against the local Module.bsl — that would invent a procedure that is not
    there.
    """
    refs = _refs(extract_edt_mdo(SUBSCRIPTION_MDO))
    assert not [r for r in refs if r[2] and r[2].startswith("service-")]


def test_bsl_links_manager_access_to_metadata():
    r = extract_bsl(EDT / "Catalogs" / "Контрагенты" / "ManagerModule.bsl")
    refs = _edge_labels(r, "references", "metadata")
    assert ("НайтиПоИНН", "Catalog.Контрагенты") in refs


def test_bsl_metadata_ref_id_matches_mdo_object_id():
    # The crux: code-side and .mdo-side produce the SAME node id, so they merge
    # into one node at build time.
    rb = extract_bsl(EDT / "Catalogs" / "Контрагенты" / "ManagerModule.bsl")
    rm = extract_edt_mdo(CATALOG_MDO)
    bsl_ids = {n["id"] for n in rb["nodes"] if n["label"] == "Catalog.Контрагенты"}
    mdo_ids = {n["id"] for n in rm["nodes"] if n["label"] == "Catalog.Контрагенты"}
    assert bsl_ids and bsl_ids == mdo_ids


# ── 1C:EDT subsystems / rights / forms ───────────────────────────────────────

SUBSYSTEM_MDO = EDT / "Subsystems" / "Продажи" / "Продажи.mdo"
RIGHTS = EDT / "Roles" / "Менеджер" / "Rights.rights"
FORM = EDT / "Catalogs" / "Контрагенты" / "Forms" / "ФормаЭлемента" / "Form.form"


def test_edt_subsystem_contains_content():
    r = extract_edt_mdo(SUBSYSTEM_MDO)
    assert "Subsystem.Продажи" in _labels(r)
    contains = _edge_labels(r, "contains")
    assert ("Subsystem.Продажи", "Catalog.Контрагенты") in contains
    assert ("Subsystem.Продажи", "Enum.СтатусДоговора") in contains


NESTED_SUBSYSTEM_MDO = (EDT / "Subsystems" / "Продажи" / "Subsystems" / "Розница"
                        / "Розница.mdo")


def test_edt_nested_subsystem_id_includes_parent_chain():
    # A nested subsystem is identified by its full parent chain, so a same-named
    # subsystem under a different parent would NOT collide onto it.
    r = extract_edt_mdo(NESTED_SUBSYSTEM_MDO)
    node = _node_by_label(r, "Subsystem.Продажи.Розница")
    assert node is not None
    assert node["id"] != _make_id("Subsystem", "Розница")   # not the bare id
    assert node["id"] == _make_id("Subsystem", "Продажи", "Розница")


def test_edt_parent_registers_nested_subsystem():
    # Parent's <subsystems>Розница</subsystems> -> contains edge whose target id
    # matches the nested subsystem's own .mdo node (they merge at build time).
    rp = extract_edt_mdo(SUBSYSTEM_MDO)
    contains = _edge_labels(rp, "contains")
    assert ("Subsystem.Продажи", "Subsystem.Продажи.Розница") in contains
    parent_child_id = _node_by_label(rp, "Subsystem.Продажи.Розница")["id"]
    nested_self_id = _node_by_label(extract_edt_mdo(NESTED_SUBSYSTEM_MDO),
                                    "Subsystem.Продажи.Розница")["id"]
    assert parent_child_id == nested_self_id


def test_edt_top_level_subsystem_id_unchanged():
    # Top-level subsystem keeps its bare id, so existing references/merge hold.
    r = extract_edt_mdo(SUBSYSTEM_MDO)
    assert _node_by_label(r, "Subsystem.Продажи")["id"] == _make_id("Subsystem", "Продажи")


def test_edt_rights_secures_granted_object():
    r = extract_edt_rights(RIGHTS)
    assert "error" not in r
    secures = _edge_labels(r, "secures")
    # Granted object -> edge; attribute-level FQN collapses onto the owning object.
    assert ("Role.Менеджер", "Catalog.Контрагенты") in secures


def test_edt_rights_skips_denied_object():
    # Catalog.Пользователи has only <value>false</value> rights -> no edge.
    r = extract_edt_rights(RIGHTS)
    secures = _edge_labels(r, "secures")
    assert ("Role.Менеджер", "Catalog.Пользователи") not in secures


def test_edt_form_references_data_objects():
    r = extract_edt_form(FORM)
    assert "error" not in r
    refs = _edge_labels(r, "references", "form")
    targets = {t for _, t in refs}
    assert "Catalog.Контрагенты" in targets      # <mainTable>
    assert "Catalog.Пользователи" in targets      # CatalogRef.Пользователи attribute


def test_edt_form_anchor_matches_mdo_form_child_id():
    # The form's reference edges hang off the SAME node the owner .mdo emits for
    # <forms><name>ФормаЭлемента</name>, so they merge at build time.
    rf = extract_edt_form(FORM)
    rm = extract_edt_mdo(CATALOG_MDO)
    anchor = rf["nodes"][0]["id"]
    mdo_form_ids = {n["id"] for n in rm["nodes"] if n["label"] == "ФормаЭлемента"}
    assert anchor in mdo_form_ids


def test_edt_form_attributes_are_nodes():
    # Form attributes become their own nodes, contained by the form, with a
    # path-based id and the form's uuid as parent_uuid (they have no own uuid).
    r = extract_edt_form(FORM)
    labels = _labels(r)
    assert "Объект" in labels
    assert "СписокДоговоров" in labels
    contains = _edge_labels(r, "contains")
    assert ("ФормаЭлемента", "Объект") in contains
    attr = _node_by_label(r, "Объект")
    assert attr["parent_uuid"] == "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    assert attr["id"] != _make_id("Объект")          # not a bare id
    assert r["nodes"][0]["uuid"] == "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"  # form anchor


def test_edt_form_handler_links_to_bsl_procedure():
    # <handlers><name>ПриСозданииНаСервере</name> -> the procedure node that
    # extract_bsl emits for the sibling Module.bsl (same id, so it merges).
    rf = extract_edt_form(FORM)
    handler_targets = {e["target"] for e in rf["edges"]
                       if e.get("context") == "form-handler"}
    rb = extract_bsl(FORM.parent / "Module.bsl")
    proc_ids = {n["id"] for n in rb["nodes"] if n["label"].strip("()") == "ПриСозданииНаСервере"}
    assert proc_ids and proc_ids <= handler_targets


DCS = (EDT / "Reports" / "ВзаиморасчетыОтчет" / "Templates" / "ОсновнаяСхема"
       / "Template.dcs")


def test_edt_dcs_references_query_tables():
    r = extract_edt_dcs(DCS)
    assert "error" not in r
    refs = _edge_labels(r, "references", "dcs")
    # Owner report -> the FROM-clause source tables (RU singular prefixes).
    assert ("Report.ВзаиморасчетыОтчет", "Catalog.Контрагенты") in refs
    assert ("Report.ВзаиморасчетыОтчет", "Document.Договор") in refs


def test_edt_dcs_fields_are_nodes():
    # DCS fields become their own nodes, contained by the owner, with a path-based
    # id and the owner object's uuid as parent_uuid.
    r = extract_edt_dcs(DCS)
    assert "ИНН" in _labels(r)
    contains = _edge_labels(r, "contains")
    assert ("Report.ВзаиморасчетыОтчет", "ИНН") in contains
    field = _node_by_label(r, "ИНН")
    assert field["parent_uuid"] == "70707070-7070-4707-8707-707070707070"


def test_edt_dcs_ignores_field_aliases():
    # `Контрагенты.ИНН` (alias.field) must NOT become a metadata reference.
    r = extract_edt_dcs(DCS)
    targets = {t for _, t in _edge_labels(r, "references", "dcs")}
    assert not any(t.endswith(".ИНН") for t in targets)

# ── 1C:EDT inline children of a .mdo ─────────────────────────────────────────

REGISTER_MDO = EDT / "InformationRegisters" / "КурсыВалют" / "КурсыВалют.mdo"
TASK_MDO = EDT / "Tasks" / "Поручение" / "Поручение.mdo"
CHART_MDO = EDT / "ChartsOfAccounts" / "Основной" / "Основной.mdo"
REPORT_MDO = (EDT / "Reports" / "ВзаиморасчетыОтчет"
              / "ВзаиморасчетыОтчет.mdo")


def test_edt_child_map_covers_every_documented_block():
    # The map is the whole mechanism: a block missing from it is a block the
    # graph never sees, and nothing else in the parser would notice.
    assert {
        "attributes", "tabularSections", "enumValues", "forms", "commands",
        "recalculations", "tableFields", "fields",
        "standardAttributes", "resources", "dimensions", "templates",
        "addressingAttributes", "accountingFlags",
        "items", "columns", "operations", "parameters", "urlTemplates",
        "methods", "integrationServiceChannels",
    } <= set(_EDT_CHILD_KINDS)


def test_edt_child_map_marks_a_skill_confirmed_subkind():
    # `InformationRegister.Курсы.Resource.Курс` is a form 1C uses itself.
    assert _EDT_CHILD_KINDS["resources"] == ("Resource", _EDT_SUBKIND_CONFIRMED)


def test_edt_child_map_marks_a_graphify_convention():
    # `HTTPService.X.URLTemplate.Y` is our spelling. 1C documents no FQN for a
    # URL template, and the graph must not imply that it does.
    assert _EDT_CHILD_KINDS["urlTemplates"] == ("URLTemplate", _EDT_SUBKIND_CONVENTION)


def test_edt_child_map_origins_are_only_the_two_known_values():
    assert {origin for _, origin in _EDT_CHILD_KINDS.values()} == {
        _EDT_SUBKIND_CONFIRMED, _EDT_SUBKIND_CONVENTION}


def test_edt_block_outside_the_map_yields_nothing(tmp_path):
    # No automatic descent: an unmapped child is ignored rather than guessed at.
    mdo = tmp_path / "Справочник.mdo"
    mdo.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mdclass:Catalog xmlns:mdclass="http://g5.1c.ru/v8/dt/metadata/mdclass"'
        ' uuid="aaaaaaaa-0000-4000-8000-000000000001">\n'
        "  <name>Справочник</name>\n"
        "  <somethingElse uuid=\"aaaaaaaa-0000-4000-8000-000000000002\">\n"
        "    <name>НеРебёнок</name>\n"
        "  </somethingElse>\n"
        "</mdclass:Catalog>\n",
        encoding="utf-8",
    )
    r = extract_edt_mdo(mdo)
    assert "НеРебёнок" not in _labels(r)
    assert _labels(r) == ["Catalog.Справочник"]


def test_edt_register_dimensions_and_resources():
    r = extract_edt_mdo(REGISTER_MDO)
    contains = _edge_labels(r, "contains")
    assert ("InformationRegister.КурсыВалют", "Валюта") in contains
    assert ("InformationRegister.КурсыВалют", "Курс") in contains
    ids = {n["id"] for n in r["nodes"]}
    assert _make_id("InformationRegister", "КурсыВалют", "Dimension", "Валюта") in ids
    assert _make_id("InformationRegister", "КурсыВалют", "Resource", "Курс") in ids


def test_edt_addressing_attribute_and_accounting_flag():
    # Neither block occurs in the audited corpus, so a fixture is the only place
    # the claim "the map covers them" can be checked against real parsing.
    assert _make_id("Task", "Поручение", "AddressingAttribute", "Исполнитель") in {
        n["id"] for n in extract_edt_mdo(TASK_MDO)["nodes"]}
    assert _make_id("ChartOfAccounts", "Основной", "AccountingFlag", "Валютный") in {
        n["id"] for n in extract_edt_mdo(CHART_MDO)["nodes"]}


def test_edt_standard_attribute_node_has_no_uuid_field():
    # The block carries no uuid attribute at all (skill §3), so the field is
    # absent rather than present and empty.
    node = _node_by_label(extract_edt_mdo(CATALOG_MDO), "Description")
    assert node is not None
    assert "uuid" not in node


def test_edt_predefined_item_id_is_kept_apart_from_uuid():
    # A predefined item's `id` is its identity in user data — a different thing
    # from a metadata uuid, and merging the two would make them indistinguishable.
    node = _node_by_label(extract_edt_mdo(CATALOG_MDO), "Основной")
    assert node["predefined_id"] == "11111111-0000-4000-8000-000000000001"
    assert "uuid" not in node


def test_edt_ordinary_blocks_still_carry_uuid():
    r = extract_edt_mdo(CATALOG_MDO)
    assert _node_by_label(r, "ИНН")["uuid"]
    assert _node_by_label(r, "ФормаЭлемента")["uuid"]


def test_edt_predefined_items_are_counted_one_by_one():
    # The container-versus-leaf trap: `<predefined>` is a wrapper with no name
    # and no uuid. Counting wrappers answers 1 for a catalog that has five
    # predefined items — measured corpus-wide as 44 wrappers over 559 items.
    r = extract_edt_mdo(CATALOG_MDO)
    predefined = [n for n in r["nodes"] if n.get("predefined_id")]
    assert len(predefined) == 5
    contains = _edge_labels(r, "contains")
    assert ("Catalog.Контрагенты", "Резервный") in contains


def test_edt_web_service_operation_parameters():
    # The parameter hangs off the operation, not off the service, and its FQN
    # carries both — the recursive grammar of skill §3.
    r = extract_edt_mdo(WEB_SERVICE_MDO)
    operation_id = _make_id("WebService", "Каталог", "Operation", "ПолучитьНоменклатуру")
    parameter_id = _make_id("WebService", "Каталог", "Operation", "ПолучитьНоменклатуру",
                            "Parameter", "Период")
    assert parameter_id in {n["id"] for n in r["nodes"]}
    assert (operation_id, parameter_id, "contains") in {
        (e["source"], e["target"], e["relation"]) for e in r["edges"]}


def test_edt_http_url_template_methods():
    r = extract_edt_mdo(HTTP_SERVICE_MDO)
    template_id = _make_id("HTTPService", "Обмен", "URLTemplate", "ШаблонЗаказа")
    method_id = _make_id("HTTPService", "Обмен", "URLTemplate", "ШаблонЗаказа",
                         "Method", "POST")
    assert (template_id, method_id, "contains") in {
        (e["source"], e["target"], e["relation"]) for e in r["edges"]}


def test_edt_child_properties_are_not_extracted():
    # Node, name, identifier and ownership — nothing else. The resource in the
    # fixture has a type with number qualifiers; none of it reaches the node.
    node = _node_by_label(extract_edt_mdo(REGISTER_MDO), "Курс")
    assert set(node) <= {"id", "label", "file_type", "source_file",
                         "source_location", "uuid", "predefined_id"}


def test_edt_template_node_does_not_re_anchor_the_dcs_edges():
    # The template now has a node of its own. The composition schema's edges
    # must still start at the report: moving them would delete existing edges,
    # which is a regression however tidy it looks.
    template_id = _make_id("Report", "ВзаиморасчетыОтчет", "Template", "ОсновнаяСхема")
    assert template_id in {n["id"] for n in extract_edt_mdo(REPORT_MDO)["nodes"]}
    dcs = extract_edt_dcs(DCS)
    assert {e["source"] for e in dcs["edges"] if e["relation"] == "references"} == {
        _make_id("Report", "ВзаиморасчетыОтчет")}


# ── 1C:EDT ordinary form (.oform) ────────────────────────────────────────────
#
# The container is built here instead of being committed as a binary fixture:
# an opaque .oform in the repository can be neither read nor regenerated. The
# writer alone does not validate the reader — the reader was measured against
# the 904 ordinary forms of a real configuration, where every file parsed, each
# held exactly the two elements `form` and `module`, and no module was split
# across blocks (578 form trees were).

_V8_NONE = 0x7FFFFFFF
_V8_STAMP = bytes.fromhex("a04f8c3c42440200") * 2 + b"\x00\x00\x00\x00"


def _v8_block(payload: bytes, doc_size: int, next_off: int, pad: int = 0) -> bytes:
    size = max(len(payload), pad)
    head = b"\r\n%08x %08x %08x \r\n" % (doc_size, size, next_off)
    return head + payload.ljust(size, b"\x00")


def _v8_document(payload: bytes, offset: int, chunk: int | None = None) -> bytes:
    """One container document, optionally split across a chain of blocks.

    `chunk` splits the payload so the continuation field (`next`) is exercised;
    a reader that stops after the first block truncates such a document.
    """
    if chunk is None or chunk >= len(payload):
        return _v8_block(payload, len(payload), _V8_NONE)
    out = b""
    for pos in range(0, len(payload), chunk):
        piece = payload[pos:pos + chunk]
        last = pos + chunk >= len(payload)
        nxt = _V8_NONE if last else offset + len(out) + 31 + len(piece)
        out += _v8_block(piece, len(payload) if pos == 0 else len(piece), nxt)
    return out


def _oform_bytes(module: bytes | None, tree: bytes = b"{1,\r\n}",
                 module_first: bool = False, chunk: int | None = None) -> bytes:
    """A V8 container of the shape an ordinary form has on disk."""
    elements = [("form", tree)]
    if module is not None:
        elements.append(("module", module))
    if module_first:
        elements.reverse()

    toc_block = 512
    cursor = 16 + 31 + toc_block
    parts: list[bytes] = []
    entries: list[tuple[int, int]] = []
    for name, payload in elements:
        head = _v8_document(_V8_STAMP + name.encode("utf-16-le") + b"\x00" * 4, cursor)
        head_off, cursor = cursor, cursor + len(head)
        body = _v8_document(payload, cursor, chunk)
        body_off, cursor = cursor, cursor + len(body)
        parts += [head, body]
        entries.append((head_off, body_off))

    toc = b"".join(h.to_bytes(4, "little") + b.to_bytes(4, "little")
                   + _V8_NONE.to_bytes(4, "little") for h, b in entries)
    return (bytes.fromhex("ffffff7f000200000200000000000000")
            + _v8_block(toc, len(toc), _V8_NONE, pad=toc_block) + b"".join(parts))


def _write_oform(tmp_path: Path, module: bytes | None, **kw) -> Path:
    """Write a form under the path shape EDT uses, so its owner resolves."""
    p = tmp_path / "Catalogs" / "Контрагенты" / "Forms" / "ФормаЭлемента" / "Form.oform"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_oform_bytes(module, **kw))
    return p


_OFORM_MODULE_EN = (
    "﻿// Ordinary form module: no compilation directives exist here.\r\n"
    "\r\n"
    "Procedure ClientOnChange(pControl)\r\n"
    "\tRecalculate();\r\n"
    "EndProcedure\r\n"
    "\r\n"
    "Function Recalculate()\r\n"
    "\tReturn ThisForm.Controls.Total;\r\n"
    "EndFunction\r\n"
).encode("utf-8")

_OFORM_MODULE_RU = (
    "﻿// Модуль обычной формы.\r\n"
    "\r\n"
    "Процедура КлиентПриИзменении(пЭлемент)\r\n"
    "\tПересчитать();\r\n"
    "КонецПроцедуры\r\n"
    "\r\n"
    "Функция Пересчитать()\r\n"
    "\tВозврат ЭтаФорма.Controls.Итого;\r\n"
    "КонецФункции\r\n"
).encode("utf-8")


def test_edt_oform_extracts_module_procedures(tmp_path):
    r = extract_edt_oform(_write_oform(tmp_path, _OFORM_MODULE_EN))
    assert "error" not in r
    assert {"ClientOnChange()", "Recalculate()"} <= set(_labels(r))


def test_edt_oform_module_calls_resolve(tmp_path):
    # The module goes through the shared extract_bsl, so its call graph is built
    # the same way a .bsl module's is.
    r = extract_edt_oform(_write_oform(tmp_path, _OFORM_MODULE_EN))
    assert ("ClientOnChange()", "Recalculate()") in _calls(r)


def test_edt_oform_nodes_indistinguishable_from_bsl(tmp_path):
    # The same module text, once inside a container and once as a plain .bsl:
    # the procedure nodes agree in every field but the file they came from.
    oform = extract_edt_oform(_write_oform(tmp_path, _OFORM_MODULE_EN))
    plain = tmp_path / "Module.bsl"
    plain.write_bytes(_OFORM_MODULE_EN)

    def procs(result):
        return {n["label"]: {k: v for k, v in n.items()
                             if k not in ("id", "source_file", "source_location")}
                for n in result["nodes"] if n["label"].endswith("()")}

    assert procs(oform) == procs(extract_bsl(plain))


def test_edt_oform_module_read_before_the_tree(tmp_path):
    # 326 of the corpus's 904 files store the module first. The element is
    # picked by name, so physical order cannot change the result.
    first = extract_edt_oform(_write_oform(tmp_path, _OFORM_MODULE_EN, module_first=True))
    second = extract_edt_oform(_write_oform(tmp_path, _OFORM_MODULE_EN))
    assert _labels(first) == _labels(second)
    assert "ClientOnChange()" in _labels(first)


def test_edt_oform_module_block_keeps_its_comment_header():
    # Taken whole, not from the first declaration: 752 modules of the corpus
    # open with a comment header, which a declaration-first recipe cuts off.
    payload, _ = _v8_elements(_oform_bytes(_OFORM_MODULE_EN))["module"]
    assert payload == _OFORM_MODULE_EN
    assert payload.startswith("﻿// Ordinary form".encode("utf-8"))


def test_edt_oform_module_spanning_blocks_is_not_truncated():
    # Container data may continue in a further block (`next`); a reader that
    # takes only the first block loses everything after it.
    payload, _ = _v8_elements(_oform_bytes(_OFORM_MODULE_EN, chunk=64))["module"]
    assert payload == _OFORM_MODULE_EN


def test_edt_oform_positions_point_into_the_form_file(tmp_path):
    # The module has no file of its own, so positions are lines of the .oform.
    path = _write_oform(tmp_path, _OFORM_MODULE_EN)
    lines = path.read_bytes().split(b"\n")
    line = int(_node_by_label(extract_edt_oform(path),
                              "ClientOnChange()")["source_location"].lstrip("L"))
    assert line > 1
    assert b"Procedure ClientOnChange" in lines[line - 1]


def test_edt_oform_file_node_stays_at_line_one(tmp_path):
    path = _write_oform(tmp_path, _OFORM_MODULE_EN)
    assert _node_by_label(extract_edt_oform(path), "Form.oform")["source_location"] == "L1"


def test_edt_oform_form_defines_its_module(tmp_path):
    # The form node already exists (emitted from the parent .mdo <forms> block);
    # the defines edge targets the .oform itself, there being no Module.bsl.
    path = _write_oform(tmp_path, _OFORM_MODULE_EN)
    r = extract_edt_oform(path)
    form_id = _make_id("Catalog", "Контрагенты", "Form", "ФормаЭлемента")
    assert (form_id, _make_id(str(path)), "defines") in {
        (e["source"], e["target"], e["relation"]) for e in r["edges"]}
    assert len([n for n in r["nodes"] if n["id"] == form_id]) == 1


def test_edt_oform_english_keywords(tmp_path):
    r = extract_edt_oform(_write_oform(tmp_path, _OFORM_MODULE_EN))
    assert {"ClientOnChange()", "Recalculate()"} <= set(_labels(r))


def test_edt_oform_russian_keywords(tmp_path):
    r = extract_edt_oform(_write_oform(tmp_path, _OFORM_MODULE_RU))
    assert {"КлиентПриИзменении()", "Пересчитать()"} <= set(_labels(r))


def test_edt_oform_single_language_scan_would_miss_a_module():
    # Why the two tests above are not redundant: this work began with the
    # measurement "no module in any of the 904 files", taken by grepping for
    # КонецПроцедуры in a configuration whose scriptVariant is English. A
    # one-language scan returns a confident, wrong zero.
    assert "КонецПроцедуры".encode("utf-8") not in _OFORM_MODULE_EN
    assert b"EndProcedure" not in _OFORM_MODULE_RU


def test_edt_oform_without_a_module(tmp_path):
    # Six of the corpus's 904 forms hold no module. The form node still stands —
    # its layout may still refer to things — but nothing is defined by it.
    r = extract_edt_oform(_write_oform(tmp_path, None))
    assert "error" not in r
    assert not [n for n in r["nodes"] if n["label"].endswith("()")]
    assert not [e for e in r["edges"] if e["relation"] == "defines"]


def test_edt_oform_with_an_empty_module(tmp_path):
    # Six of the corpus's forms hold a module element with nothing in it. There
    # is no file node to define, so no `defines` edge may claim one.
    r = extract_edt_oform(_write_oform(tmp_path, b""))
    assert "error" not in r
    ids = {n["id"] for n in r["nodes"]}
    for edge in r["edges"]:
        assert edge["target"] in ids, f"dangling target: {edge}"


def test_edt_oform_empty_file(tmp_path):
    path = _write_oform(tmp_path, _OFORM_MODULE_EN)
    path.write_bytes(b"")
    r = extract_edt_oform(path)
    assert r["error"] and r["nodes"] == [] and r["edges"] == []


def test_edt_oform_truncated_file(tmp_path):
    path = _write_oform(tmp_path, _OFORM_MODULE_EN)
    path.write_bytes(path.read_bytes()[:100])
    r = extract_edt_oform(path)
    assert r["error"] and r["nodes"] == [] and r["edges"] == []


def test_edt_oform_not_a_container(tmp_path):
    path = _write_oform(tmp_path, _OFORM_MODULE_EN)
    path.write_bytes(b"this is not a V8 container" * 40)
    r = extract_edt_oform(path)
    assert r["error"] and r["nodes"] == [] and r["edges"] == []


def test_edt_oform_is_never_written_to(tmp_path):
    # The file is binary and EDT does not open it; only the Configurator edits
    # one. Extraction leaves it byte for byte as it was.
    import hashlib

    path = _write_oform(tmp_path, _OFORM_MODULE_EN)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    extract_edt_oform(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


# ── 1C:EDT ordinary form: references out of the layout ───────────────────────
#
# The layout names nothing it refers to — it carries uuids, and only the .mdo
# files say what they are. Measured across the corpus: the `Kind.Name` tokens
# the design first expected here occur exactly zero times in 904 layouts, while
# 899 of those files hold at least one uuid that a .mdo declares.

_OBJECT_UUID = "b39b8b42-0061-4aac-a0de-206ebc513617"
_ATTRIBUTE_UUID = "c1c1c1c1-0000-4000-8000-000000000002"
_FORM_UUID = "c1c1c1c1-0000-4000-8000-000000000003"
_REF_TYPE_UUID = "e9a9379f-c522-4ea0-87d6-6d9791afdd61"
_UNKNOWN_UUID = "deadbeef-0000-4000-8000-000000000009"

_OFORM_MDO = f"""<?xml version="1.0" encoding="UTF-8"?>
<mdclass:Catalog xmlns:mdclass="http://g5.1c.ru/v8/dt/metadata/mdclass"
    uuid="{_OBJECT_UUID}">
  <producedTypes>
    <refType typeId="{_REF_TYPE_UUID}" valueTypeId="f8c75947-3d64-477c-a11d-b9d8a665c2e5"/>
  </producedTypes>
  <name>Контрагенты</name>
  <attributes uuid="{_ATTRIBUTE_UUID}">
    <name>ИНН</name>
  </attributes>
  <forms uuid="{_FORM_UUID}">
    <name>ФормаЭлемента</name>
  </forms>
</mdclass:Catalog>
"""

_OFORM_FORM_ID = _make_id("Catalog", "Контрагенты", "Form", "ФормаЭлемента")


def _write_oform_project(tmp_path: Path, tree: bytes,
                         module: bytes | None = _OFORM_MODULE_EN) -> Path:
    """A one-object project: the .mdo that declares the uuids, plus the form."""
    owner = tmp_path / "Catalogs" / "Контрагенты"
    (owner / "Forms" / "ФормаЭлемента").mkdir(parents=True, exist_ok=True)
    (owner / "Контрагенты.mdo").write_text(_OFORM_MDO, encoding="utf-8")
    form = owner / "Forms" / "ФормаЭлемента" / "Form.oform"
    form.write_bytes(_oform_bytes(module, tree=tree))
    return form


def _oform_refs(result: dict) -> set[str]:
    labels = {n["id"]: n["label"] for n in result["nodes"]}
    return {labels.get(e["target"], e["target"]) for e in result["edges"]
            if e["relation"] == "references" and e.get("context") == "oform"}


def test_oform_layout_token_kinds():
    # One token per construct of the bracket notation, doubled quote included:
    # `""` inside a string is a quote, not the end of the string.
    sample = '{27,"a,b","say ""{hi}""",09ccdc77-ea1a-4a6d-ab1c-3435eada2433,-1.5,#}'
    assert _OFORM_TOKEN_RE.findall(sample) == [
        "{", "27", ",", '"a,b"', ",", '"say ""{hi}"""', ",",
        "09ccdc77-ea1a-4a6d-ab1c-3435eada2433", ",", "-1.5", ",", "#", "}",
    ]


def test_oform_uuid_inside_a_string_is_not_a_reference():
    # A uuid written into a caption or a stored setting is data. Tokenising is
    # what tells the two apart; a regex over the file cannot.
    assert _oform_tree_uuids(f'{{1,"{_OBJECT_UUID}"}}') == []
    assert _oform_tree_uuids(f"{{1,{_OBJECT_UUID}}}") == [_OBJECT_UUID]


def test_oform_uuid_resolves_to_the_object(tmp_path):
    form = _write_oform_project(tmp_path, f"{{1,{_OBJECT_UUID}}}".encode("utf-8"))
    assert "Catalog.Контрагенты" in _oform_refs(extract_edt_oform(form))


def test_oform_uuid_resolves_to_an_attribute(tmp_path):
    # The valuable half: which attribute a control is bound to. 689 files of the
    # corpus carry attribute uuids.
    form = _write_oform_project(tmp_path, f"{{1,{_ATTRIBUTE_UUID}}}".encode("utf-8"))
    result = extract_edt_oform(form)
    assert "ИНН" in _oform_refs(result)
    assert _make_id("Catalog", "Контрагенты", "Attribute", "ИНН") in {
        e["target"] for e in result["edges"] if e.get("context") == "oform"}


def test_oform_produced_type_resolves_to_its_object(tmp_path):
    # A control typed `CatalogRef.Контрагенты` carries the type's uuid, not the
    # object's; 883 of the corpus's 904 forms reach an object that way.
    form = _write_oform_project(tmp_path, f"{{1,{_REF_TYPE_UUID}}}".encode("utf-8"))
    assert "Catalog.Контрагенты" in _oform_refs(extract_edt_oform(form))


def test_oform_unknown_uuid_gives_no_edge(tmp_path):
    # Nothing is invented: a uuid no .mdo declares is not a reference. The index
    # is what replaced the kind whitelist.
    form = _write_oform_project(tmp_path, f"{{1,{_UNKNOWN_UUID}}}".encode("utf-8"))
    assert _oform_refs(extract_edt_oform(form)) == set()


def test_oform_caption_with_a_dot_gives_no_edge(tmp_path):
    # `Listendruck.Print`, `Settings.X` and friends: interface text of a
    # localised configuration, and the reason a `Kind.Name` whitelist was
    # specified in the first place. Nothing textual becomes a reference now.
    tree = '{1,"Listendruck.Print","Catalog.Контрагенты","Settings.X"}'.encode("utf-8")
    form = _write_oform_project(tmp_path, tree)
    assert _oform_refs(extract_edt_oform(form)) == set()


def test_oform_uuid_outside_the_layout_is_not_a_reference(tmp_path):
    # Only the layout element is read. A uuid in the module text — or in the
    # container's service bytes — is outside the stream that carries references,
    # even though a scan of the raw file would find it.
    module = (_OFORM_MODULE_EN.decode("utf-8")
              + f'\r\nProcedure Note()\r\n\tX = "{_OBJECT_UUID}";\r\nEndProcedure\r\n'
              ).encode("utf-8")
    form = _write_oform_project(tmp_path, b"{1,\r\n}", module=module)
    assert _OBJECT_UUID.encode("utf-8") in form.read_bytes()
    assert _oform_refs(extract_edt_oform(form)) == set()


def test_oform_does_not_reference_itself(tmp_path):
    # The layout carries the form's own uuid. An edge from the form to itself is
    # not a fact about anything.
    form = _write_oform_project(tmp_path, f"{{1,{_FORM_UUID}}}".encode("utf-8"))
    result = extract_edt_oform(form)
    assert _oform_refs(result) == set()
    assert not [e for e in result["edges"] if e["source"] == e["target"]]


def test_oform_references_start_from_the_form_node_the_mdo_emits(tmp_path):
    # Anchored on the node the parent .mdo already created for this form, so the
    # two collapse into one instead of splitting the form in half.
    form = _write_oform_project(tmp_path, f"{{1,{_OBJECT_UUID}}}".encode("utf-8"))
    result = extract_edt_oform(form)
    mdo = extract_edt_mdo(form.parent.parent.parent / "Контрагенты.mdo")
    assert {e["source"] for e in result["edges"]
            if e.get("context") == "oform"} == {_OFORM_FORM_ID}
    assert _OFORM_FORM_ID in {n["id"] for n in mdo["nodes"]}


def test_oform_reference_edges_are_not_dangling(tmp_path):
    form = _write_oform_project(
        tmp_path,
        f"{{1,{_OBJECT_UUID},{_ATTRIBUTE_UUID},{_UNKNOWN_UUID}}}".encode("utf-8"))
    result = extract_edt_oform(form)
    ids = {n["id"] for n in result["nodes"]}
    for edge in result["edges"]:
        assert edge["source"] in ids, f"dangling source: {edge}"
        if edge["relation"] != "imports":
            assert edge["target"] in ids, f"dangling target: {edge}"


def test_oform_without_a_module_still_yields_references(tmp_path):
    form = _write_oform_project(tmp_path, f"{{1,{_OBJECT_UUID}}}".encode("utf-8"),
                                module=None)
    result = extract_edt_oform(form)
    assert "Catalog.Контрагенты" in _oform_refs(result)
    assert not [n for n in result["nodes"] if n["label"].endswith("()")]


# ── 1C:EDT ordinary form: controls out of the layout ─────────────────────────
#
# The layout grammar is positional and undocumented. What is read here is the
# one record shape measurement pinned down — `{14,"<Name>",…` opens a named
# control — and the change kept it only because the measurement cleared the
# threshold it set in advance: names come out of 100% of the corpus's 904 forms.

_OFORM_LAYOUT = ('{27,\r\n{14,"ПанельКнопок",4294967295,0,0,0},\r\n'
                 '{14,"КнопкаЗакрыть",4294967295,0,0,0},\r\n'
                 '{18,"НеЭлемент"}\r\n}').encode("utf-8")


def _oform_contains(result: dict) -> set[str]:
    labels = {n["id"]: n["label"] for n in result["nodes"]}
    return {labels.get(e["target"], e["target"]) for e in result["edges"]
            if e["relation"] == "contains" and e["source"] == _OFORM_FORM_ID}


def test_oform_controls_become_nodes(tmp_path):
    form = _write_oform_project(tmp_path, _OFORM_LAYOUT)
    assert {"ПанельКнопок", "КнопкаЗакрыть"} <= _oform_contains(extract_edt_oform(form))


def test_oform_control_id_hangs_off_the_form(tmp_path):
    # Same scheme a managed form's attributes use: the control has no uuid of
    # its own, so the id is the form's id plus the name.
    form = _write_oform_project(tmp_path, _OFORM_LAYOUT)
    ids = {n["id"] for n in extract_edt_oform(form)["nodes"]}
    assert _make_id(_OFORM_FORM_ID, "FormElement", "ПанельКнопок") in ids


def test_oform_control_records_the_form_uuid(tmp_path):
    form = _write_oform_project(tmp_path, _OFORM_LAYOUT)
    node = _node_by_label(extract_edt_oform(form), "КнопкаЗакрыть")
    assert node["parent_uuid"] == _FORM_UUID


def test_oform_other_records_are_not_controls(tmp_path):
    # Only the record that measurement identified is read. `{18,"…"` is some
    # other structure, and guessing that every `{N,"…"` names a control would
    # populate the graph with whatever else the layout stores.
    form = _write_oform_project(tmp_path, _OFORM_LAYOUT)
    assert "НеЭлемент" not in _oform_contains(extract_edt_oform(form))


def test_oform_control_names_keep_layout_order():
    assert _oform_element_names(_OFORM_LAYOUT.decode("utf-8")) == [
        "ПанельКнопок", "КнопкаЗакрыть"]


# ── 1C:EDT command interface (.cmi) ──────────────────────────────────────────

CONFIG_CMI = EDT / "Configuration" / "CommandInterface.cmi"
SUBSYSTEM_CMI = EDT / "Subsystems" / "Продажи" / "CommandInterface.cmi"
NESTED_CMI = (EDT / "Subsystems" / "Продажи" / "Subsystems" / "Розница"
              / "CommandInterface.cmi")


def test_edt_cmi_no_error():
    assert "error" not in extract_edt_cmi(SUBSYSTEM_CMI)


def test_edt_cmi_owner_is_the_configuration():
    r = extract_edt_cmi(CONFIG_CMI)
    assert {e["source"] for e in r["edges"]} == {_make_id("Configuration")}


def test_edt_cmi_owner_is_the_subsystem():
    r = extract_edt_cmi(SUBSYSTEM_CMI)
    assert {e["source"] for e in r["edges"]} == {_make_id("Subsystem", "Продажи")}


def test_edt_cmi_owner_of_a_nested_subsystem_carries_the_chain():
    # The same id the nested subsystem's own .mdo emits — otherwise the command
    # interface would hang off a second, parallel node for the same subsystem.
    r = extract_edt_cmi(NESTED_CMI)
    owner = _make_id("Subsystem", "Продажи", "Розница")
    assert {e["source"] for e in r["edges"]} == {owner}
    assert owner in {n["id"] for n in extract_edt_mdo(NESTED_SUBSYSTEM_MDO)["nodes"]}


def test_edt_cmi_references_commands_and_roles():
    refs = _edge_labels(extract_edt_cmi(SUBSYSTEM_CMI), "references", "command-interface")
    assert ("Subsystem.Продажи", "CommonCommand.ОбщаяКоманда") in refs
    assert ("Subsystem.Продажи", "Role.Менеджер") in refs


def test_edt_cmi_object_command_keeps_its_own_node():
    # `Catalog.Контрагенты.Command.Команда` is a metadata object in its own
    # right, and the .mdo emits a node for it under exactly this id.
    r = extract_edt_cmi(SUBSYSTEM_CMI)
    command_id = _make_id("Catalog", "Контрагенты", "Command", "Печать")
    assert command_id in {e["target"] for e in r["edges"]}
    assert command_id in {n["id"] for n in extract_edt_mdo(CATALOG_MDO)["nodes"]}


def test_edt_cmi_standard_command_lands_on_its_object():
    # A standard command is provided by the platform and has no node of its own;
    # 1217 of the corpus's 1653 values are of this shape, and inventing a node
    # for each would populate the graph with metadata that does not exist.
    r = extract_edt_cmi(SUBSYSTEM_CMI)
    targets = {e["target"] for e in r["edges"]}
    assert _make_id("Catalog", "Контрагенты") in targets
    assert _make_id("Catalog", "Контрагенты", "StandardCommand", "OpenList") not in targets


def test_edt_cmi_nested_subsystem_value_collapses_the_repeated_marker():
    # `Subsystem.Продажи.Subsystem.Розница` is the FQN spelling; the node id
    # keeps the chain without repeating the kind.
    r = extract_edt_cmi(CONFIG_CMI)
    assert _make_id("Subsystem", "Продажи", "Розница") in {e["target"] for e in r["edges"]}


def test_edt_cmi_unknown_kind_gives_no_edge():
    # `Listendruck.Печать` — interface text of a localised configuration.
    r = extract_edt_cmi(SUBSYSTEM_CMI)
    assert not [e for e in r["edges"] if "Listendruck" in e["target"]]


def test_edt_cmi_does_not_reference_its_own_owner():
    # The subsystem lists itself in <subsystemsVisibility>.
    r = extract_edt_cmi(SUBSYSTEM_CMI)
    assert not [e for e in r["edges"] if e["source"] == e["target"]]


def test_edt_cmi_no_dangling_edges():
    for cmi in (CONFIG_CMI, SUBSYSTEM_CMI, NESTED_CMI):
        r = extract_edt_cmi(cmi)
        ids = {n["id"] for n in r["nodes"]}
        for edge in r["edges"]:
            assert edge["source"] in ids and edge["target"] in ids, f"dangling: {edge}"


# ── 1C:EDT project kinds (configuration / extension / external objects) ───
#
# One workspace holds all three side by side. The fixture tree mirrors that:
# `base` extends nothing, `ext` adopts `base`'s catalog under the same name, and
# `external` is a standalone data processor with no configuration root at all.

PROJECTS = FIXTURES / "edt_projects"
BASE_PROJECT = PROJECTS / "base"
EXT_PROJECT = PROJECTS / "ext"
EXTERNAL_PROJECT = PROJECTS / "external"

BASE_CATALOG_MDO = BASE_PROJECT / "src/Catalogs/Контрагенты/Контрагенты.mdo"
BASE_CONFIG_MDO = BASE_PROJECT / "src/Configuration/Configuration.mdo"
EXT_ADOPTED_MDO = EXT_PROJECT / "src/Catalogs/Контрагенты/Контрагенты.mdo"
EXT_NATIVE_MDO = EXT_PROJECT / "src/Catalogs/Расш_Своя/Расш_Своя.mdo"
EXT_CONFIG_MDO = EXT_PROJECT / "src/Configuration/Configuration.mdo"
EXT_RIGHTS = EXT_PROJECT / "src/Roles/Расш_Роль/Rights.rights"
EXTERNAL_MDO = (EXTERNAL_PROJECT
                / "src/ExternalDataProcessors/ВнешняяОбработка/ВнешняяОбработка.mdo")
EXTERNAL_FORM = (EXTERNAL_PROJECT / "src/ExternalDataProcessors/ВнешняяОбработка"
                 / "Forms/Форма/Form.form")

EXT_SCOPE = ("Extension", "Расш")


def _ids(result: dict) -> set[str]:
    return {node["id"] for node in result["nodes"]}


def _node(result: dict, node_id: str) -> dict:
    for node in result["nodes"]:
        if node["id"] == node_id:
            return node
    raise AssertionError(f"missing node id {node_id!r}")


# --- project kind -------------------------------------------------------------

def test_edt_project_kind_reads_all_three_natures():
    assert _edt_project_kind(BASE_PROJECT)[0] == EDT_PROJECT_CONFIGURATION
    assert _edt_project_kind(EXT_PROJECT)[0] == EDT_PROJECT_EXTENSION
    assert _edt_project_kind(EXTERNAL_PROJECT)[0] == EDT_PROJECT_EXTERNAL_OBJECTS


def test_edt_project_name_comes_from_the_project_file():
    assert _edt_project_kind(EXT_PROJECT)[1] == "Расш"


def test_edt_project_kind_is_one_per_project_not_per_file():
    """The adopted and the native object of one extension share a project kind.

    Reading the kind from the .mdo instead would answer differently for the two:
    only the adopted one carries the adoption markers.
    """
    assert _edt_id_scope(EXT_ADOPTED_MDO.parent) == EXT_SCOPE
    assert _edt_id_scope(EXT_NATIVE_MDO.parent) == EXT_SCOPE


def test_edt_no_project_file_behaves_as_ordinary_configuration():
    """The fixture configuration tree has no .project — ids must not move."""
    assert _edt_id_scope(CATALOG_MDO.parent) == ()
    assert _make_id("Catalog", "Контрагенты") in _ids(extract_edt_mdo(CATALOG_MDO))


def test_edt_external_objects_project_is_not_scoped():
    assert _edt_id_scope(EXTERNAL_MDO.parent) == ()


# --- ids: base and extension no longer collapse -------------------------------

def test_edt_extension_object_does_not_collide_with_the_base_object():
    base = extract_edt_mdo(BASE_CATALOG_MDO)
    ext = extract_edt_mdo(EXT_ADOPTED_MDO)
    base_obj = _make_id("Catalog", "Контрагенты")
    ext_obj = _make_id(*EXT_SCOPE, "Catalog", "Контрагенты")
    assert base_obj != ext_obj
    assert base_obj in _ids(base) and ext_obj in _ids(ext)
    # The only id the two share is the base object the extension adopts, and it
    # is present in the extension's result as the target of that one edge.
    assert _ids(base) & _ids(ext) == {base_obj}


def test_edt_ordinary_configuration_ids_are_unchanged():
    """Pinned literals, not values recomputed from the code under test.

    Recomputing would make the assertion true by construction — exactly the way
    an id shift goes unnoticed.
    """
    ids = _ids(extract_edt_mdo(CATALOG_MDO))
    assert "catalog_контрагенты" in ids
    assert "catalog_контрагенты_attribute_инн" in ids
    assert "configuration" in _ids(extract_edt_mdo(CONFIG_MDO))


def test_edt_extension_configuration_root_is_scoped():
    ext_conf = _make_id(*EXT_SCOPE, "Configuration")
    assert ext_conf in _ids(extract_edt_mdo(EXT_CONFIG_MDO))
    assert "configuration" in _ids(extract_edt_mdo(BASE_CONFIG_MDO))


def test_edt_extension_configuration_registers_its_own_objects_in_scope():
    r = extract_edt_mdo(EXT_CONFIG_MDO)
    contains = {e["target"] for e in r["edges"] if e["relation"] == "contains"}
    assert _make_id(*EXT_SCOPE, "Catalog", "Контрагенты") in contains
    assert _make_id(*EXT_SCOPE, "Catalog", "Расш_Своя") in contains
    assert _make_id(*EXT_SCOPE, "Role", "Расш_Роль") in contains


def test_edt_extension_registration_without_its_own_mdo_stays_in_the_base(tmp_path):
    """A registration the project does not back with a .mdo is a base reference.

    The scoped id claims "this project defines it"; the claim is checked against
    the disk rather than assumed from the registration list.
    """
    project = tmp_path / "Расш2"
    (project / "src" / "Configuration").mkdir(parents=True)
    (project / ".project").write_text(
        "<projectDescription><name>Расш2</name><natures>"
        "<nature>com._1c.g5.v8.dt.core.V8ExtensionNature</nature>"
        "</natures></projectDescription>", encoding="utf-8")
    (project / "src" / "Configuration" / "Configuration.mdo").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<mdclass:Configuration xmlns:mdclass="http://g5.1c.ru/v8/dt/metadata/mdclass">'
        "<name>Расш2</name><catalogs>Catalog.НетТакого</catalogs>"
        "</mdclass:Configuration>", encoding="utf-8")
    r = extract_edt_mdo(project / "src" / "Configuration" / "Configuration.mdo")
    contains = {e["target"] for e in r["edges"] if e["relation"] == "contains"}
    assert contains == {_make_id("Catalog", "НетТакого")}


# --- belonging: an attribute, never a kind ------------------------------------

def test_edt_adopted_object_is_marked_and_keeps_its_kind():
    node = _node(extract_edt_mdo(EXT_ADOPTED_MDO),
                 _make_id(*EXT_SCOPE, "Catalog", "Контрагенты"))
    assert node["object_belonging"] == "Adopted"
    assert node["label"] == "Extension.Расш.Catalog.Контрагенты"


def test_edt_native_extension_object_is_marked_native():
    node = _node(extract_edt_mdo(EXT_NATIVE_MDO),
                 _make_id(*EXT_SCOPE, "Catalog", "Расш_Своя"))
    assert node["object_belonging"] == "Native"


def test_edt_ordinary_configuration_object_carries_no_belonging():
    assert "object_belonging" not in _node(extract_edt_mdo(BASE_CATALOG_MDO),
                                           _make_id("Catalog", "Контрагенты"))


def test_edt_adoption_introduces_no_pseudo_kind():
    """`Catalog` stays `Catalog`: the kind set must not grow for adopted objects."""
    assert not [k for k in _EDT_KIND_PREFIXES
                if k.startswith("Adopted") or k.startswith("Native")]
    for result in (extract_edt_mdo(EXT_ADOPTED_MDO), extract_edt_mdo(EXT_NATIVE_MDO)):
        for node_id in _ids(result):
            assert "adopted" not in node_id and "native" not in node_id


# --- the adopted -> base link -------------------------------------------------

def test_edt_adopted_object_references_the_base_object():
    r = extract_edt_mdo(EXT_ADOPTED_MDO)
    assert (_make_id(*EXT_SCOPE, "Catalog", "Контрагенты"),
            _make_id("Catalog", "Контрагенты")) in {
        (e["source"], e["target"]) for e in r["edges"]
        if e["relation"] == "references" and e.get("context") == "adopted-from"}


def test_edt_adopted_object_links_without_the_base_project():
    """Extraction of the extension alone still yields the link, via a stub."""
    r = extract_edt_mdo(EXT_ADOPTED_MDO)
    assert _node(r, _make_id("Catalog", "Контрагенты"))["label"] == "Catalog.Контрагенты"


def test_edt_native_object_has_no_adoption_link():
    r = extract_edt_mdo(EXT_NATIVE_MDO)
    assert not [e for e in r["edges"] if e.get("context") == "adopted-from"]


# --- scope boundary: values keep base ids -------------------------------------

def test_edt_extension_rights_secure_base_objects():
    """The role is the extension's; the objects it secures are the base's.

    Measured on a real workspace: one extension role secures 4 613 objects of the
    configuration it extends. Scoping those would invent 4 613 objects.
    """
    r = extract_edt_rights(EXT_RIGHTS)
    assert [(e["source"], e["target"]) for e in r["edges"]] == [
        (_make_id(*EXT_SCOPE, "Role", "Расш_Роль"),
         _make_id("Catalog", "Контрагенты"))]


# --- external-objects projects ------------------------------------------------

def test_edt_external_object_kinds_are_recognised():
    assert {"ExternalDataProcessor", "ExternalReport"} <= _EDT_KIND_PREFIXES
    assert _EDT_PLURAL_TO_KIND["ExternalDataProcessors"] == "ExternalDataProcessor"
    assert _EDT_PLURAL_TO_KIND["ExternalReports"] == "ExternalReport"


def test_edt_external_processor_owns_its_form_and_module():
    r = extract_edt_mdo(EXTERNAL_MDO)
    obj = _make_id("ExternalDataProcessor", "ВнешняяОбработка")
    form = _make_id("ExternalDataProcessor", "ВнешняяОбработка", "Form", "Форма")
    assert (obj, form) in {(e["source"], e["target"]) for e in r["edges"]
                           if e["relation"] == "contains"}
    defines = {e["target"] for e in r["edges"] if e["relation"] == "defines"}
    assert any(t.endswith("objectmodule_bsl") for t in defines)


def test_edt_external_processor_form_content_is_extracted():
    """Without the folder in the map this file resolved no owner and was dropped."""
    r = extract_edt_form(EXTERNAL_FORM)
    form = _make_id("ExternalDataProcessor", "ВнешняяОбработка", "Form", "Форма")
    assert form in _ids(r)
    assert _make_id("Catalog", "Контрагенты") in _ids(r)


def test_edt_external_objects_project_needs_no_configuration_root():
    """No Configuration.mdo exists here by construction, and that is not an error."""
    assert not (EXTERNAL_PROJECT / "src" / "Configuration").exists()
    assert "error" not in extract_edt_mdo(EXTERNAL_MDO)
    assert "error" not in extract_edt_form(EXTERNAL_FORM)


# ── Header/impl class merge + .h routing (#1547 C++, #1556 ObjC/Swift) ─────────
from graphify.extract import (
    extract as _extract_corpus,
    _get_extractor,
    _is_cpp_header,
    _is_objc_header,
)


def _corpus(*relpaths):
    """Run the full extract() pipeline on fixture files (absolute, resolved
    paths so the per-file id-remap behaves like real usage), no shared cache."""
    import tempfile
    paths = [(FIXTURES / rp).resolve() for rp in relpaths]
    with tempfile.TemporaryDirectory() as td:
        return _extract_corpus(paths, cache_root=Path(td))


def _nodes_with_label(r, label):
    return [n for n in r["nodes"] if n["label"] == label]


def _assert_no_dangling(r):
    ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in ids, f"dangling source: {e}"
        assert e["target"] in ids, f"dangling target: {e}"


# --- #1547: C++ paired header/impl --------------------------------------------

def test_cpp_header_routes_to_cpp_extractor():
    """A `.h` with a C++ class must route to extract_cpp, not extract_c (which has
    no class_specifier and would drop the class entirely)."""
    p = (FIXTURES / "cpp_paired" / "Foo.h").resolve()
    assert _get_extractor(p).__name__ == "extract_cpp"
    assert _is_cpp_header(p)


def test_plain_c_header_stays_on_c_extractor():
    """A plain C header (no C++ signal) must keep its extract_c routing."""
    p = (FIXTURES / "cpp_samedir" / "plain.h").resolve()
    assert not _is_cpp_header(p)
    assert _get_extractor(p).__name__ == "extract_c"


def test_cpp_paired_single_class_node():
    """Foo.h (class) + Foo.cpp (Foo::bar def) + Main.cpp must yield exactly ONE
    Foo class node — not a foo_h + foo_cpp pair, and no junk `class` stub."""
    r = _corpus("cpp_paired/Foo.h", "cpp_paired/Foo.cpp", "cpp_paired/Main.cpp")
    foos = _nodes_with_label(r, "Foo")
    assert len(foos) == 1, f"expected one Foo, got {[n['id'] for n in foos]}"
    assert not _nodes_with_label(r, "class"), "no sourceless `class` stub should exist"
    assert not _nodes_with_label(r, "foo_foo")


def test_cpp_paired_method_decl_and_def_are_one_node():
    """`void bar();` in Foo.h and `void Foo::bar() {}` in Foo.cpp must collapse to
    ONE method node owned by the single Foo class."""
    r = _corpus("cpp_paired/Foo.h", "cpp_paired/Foo.cpp", "cpp_paired/Main.cpp")
    foo = _nodes_with_label(r, "Foo")[0]["id"]
    method_targets = {
        e["target"] for e in r["edges"]
        if e["source"] == foo and e["relation"] in ("method", "defines", "contains")
    }
    bar_nodes = [n for n in r["nodes"] if n["id"] in method_targets and n["label"] in ("bar", "Foo::bar()")]
    # There must be exactly one node representing bar (decl and def merged).
    bar_ids = {n["id"] for n in r["nodes"] if n["label"] in ("bar", "Foo::bar()")}
    assert len(bar_ids) == 1, f"bar decl/def should be one node, got {bar_ids}"
    assert bar_nodes, "the merged bar node should be a member of Foo"


def test_cpp_paired_includes_resolve_to_real_header():
    """Foo.cpp and Main.cpp `#include "Foo.h"` must resolve to the real Foo.h file
    node (no dangling import)."""
    r = _corpus("cpp_paired/Foo.h", "cpp_paired/Foo.cpp", "cpp_paired/Main.cpp")
    ids = {n["id"] for n in r["nodes"]}
    foo_h = _nodes_with_label(r, "Foo.h")[0]["id"]
    imports = [e for e in r["edges"] if e["relation"] == "imports"]
    assert len(imports) >= 2
    for e in imports:
        assert e["target"] in ids, f"dangling import target: {e}"
    assert any(e["target"] == foo_h for e in imports), "includes should target Foo.h"


def test_cpp_paired_no_dangling_edges():
    r = _corpus("cpp_paired/Foo.h", "cpp_paired/Foo.cpp", "cpp_paired/Main.cpp")
    _assert_no_dangling(r)


# --- #1556: ObjC paired header/impl + bridging header -------------------------

def test_objc_header_with_import_routes_to_objc():
    """A bridging header that is only `#import "X.h"` (no @interface) must route to
    extract_objc; extract_c parses `#import` as preproc_call and drops the edge."""
    p = (FIXTURES / "objc_mixed" / "Bridging-Header.h").resolve()
    assert _is_objc_header(p)
    assert _get_extractor(p).__name__ == "extract_objc"


def test_objc_paired_single_class_methods_not_duplicated():
    """Widget.h (@interface) + Widget.m (@implementation) -> ONE Widget class node
    with its methods present once each."""
    r = _corpus("objc_mixed/Widget.h", "objc_mixed/Widget.m")
    widgets = _nodes_with_label(r, "Widget")
    assert len(widgets) == 1, f"expected one Widget, got {[n['id'] for n in widgets]}"
    render = _nodes_with_label(r, "-render")
    refresh = _nodes_with_label(r, "-refresh")
    assert len(render) == 1, f"-render duplicated: {render}"
    assert len(refresh) == 1, f"-refresh duplicated: {refresh}"


def test_objc_bridging_header_not_isolated():
    """A bridging header of only `#import "Widget.h"` must produce an imports edge
    to the real Widget.h node (not be an isolated node)."""
    r = _corpus("objc_mixed/Widget.h", "objc_mixed/Widget.m", "objc_mixed/Bridging-Header.h")
    bridge = _nodes_with_label(r, "Bridging-Header.h")[0]["id"]
    widget_h = _nodes_with_label(r, "Widget.h")[0]["id"]
    out = [e for e in r["edges"] if e["source"] == bridge and e["relation"] == "imports"]
    assert out, "bridging header should emit an imports edge"
    assert any(e["target"] == widget_h for e in out), "bridging import should target Widget.h"


def test_objc_paired_no_dangling_edges():
    r = _corpus("objc_mixed/Widget.h", "objc_mixed/Widget.m", "objc_mixed/Bridging-Header.h")
    _assert_no_dangling(r)


# --- #1556: Swift extension folds onto canonical ObjC class -------------------

def test_swift_extension_folds_onto_objc_class():
    """`extension Widget` in Swift over an ObjC `Widget` must fold onto the single
    canonical Widget node, with its members anchored there."""
    r = _corpus("objc_mixed/Widget.h", "objc_mixed/Widget.m", "objc_mixed/WidgetExtras.swift")
    widgets = _nodes_with_label(r, "Widget")
    assert len(widgets) == 1, f"expected one Widget, got {[n['id'] for n in widgets]}"
    wid = widgets[0]["id"]
    method_targets = {e["target"] for e in r["edges"] if e["relation"] == "method" and e["source"] == wid}
    labels = {n["label"] for n in r["nodes"] if n["id"] in method_targets}
    assert any("describe" in l for l in labels), f"Swift extension method should anchor on Widget, got {labels}"
    _assert_no_dangling(r)


# --- god-node guard negatives -------------------------------------------------

def test_decldef_merge_does_not_merge_across_directories():
    """Two unrelated `class Logger` in DIFFERENT directories (each its own .h/.cpp)
    must NOT merge — assert TWO distinct Logger nodes."""
    r = _corpus(
        "cpp_logger/a/Logger.h", "cpp_logger/a/Logger.cpp",
        "cpp_logger/b/Logger.h", "cpp_logger/b/Logger.cpp",
    )
    loggers = _nodes_with_label(r, "Logger")
    assert len(loggers) == 2, f"cross-dir Loggers must stay distinct, got {[n['id'] for n in loggers]}"
    assert len({n["id"] for n in loggers}) == 2


def test_decldef_merge_does_not_merge_same_name_same_dir_distinct_files():
    """Two same-named `class Dup` in the SAME dir but different base stems
    (Alpha.h, Beta.h) must stay distinct (no unique header/impl sibling pair)."""
    r = _corpus("cpp_samedir/Alpha.h", "cpp_samedir/Beta.h")
    dups = _nodes_with_label(r, "Dup")
    assert len(dups) == 2, f"same-dir distinct Dups must stay distinct, got {[n['id'] for n in dups]}"
