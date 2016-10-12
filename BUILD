package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = ["//NerveSegmentation/..."],
)

py_library(
    name = "dataset",
    srcs = [
        "dataset.py",
    ],
)

py_library(
    name = "nerve_data",
    srcs = [
        "nerve_data.py",
    ],
    deps = [
        ":dataset",
    ],
)

py_binary(
    name = "build_dataset",
    srcs = ["build_dataset.py"],
)

filegroup(
    name = "srcs",
    srcs = glob(
        [
            "**/*.py",
            "BUILD",
        ],
    ),
)
