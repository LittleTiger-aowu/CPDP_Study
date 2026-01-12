# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'my-languages.so',

  # Include one or more languages
  [

    'F:\\PycharmProject\\CodeBERT+GraphCodeBert+UniXcoder\\GraphCodeBERT\\clonedetection\\parser\\tree-sitter-go',
    'F:\\PycharmProject\\CodeBERT+GraphCodeBert+UniXcoder\\GraphCodeBERT\\clonedetection\\parser\\tree-sitter-javascript',
    'F:\\PycharmProject\\CodeBERT+GraphCodeBert+UniXcoder\\GraphCodeBERT\\clonedetection\\parser\\tree-sitter-python',
    'F:\\PycharmProject\\CodeBERT+GraphCodeBert+UniXcoder\\GraphCodeBERT\\clonedetection\\parser\\tree-sitter-php',
    'F:\\PycharmProject\\CodeBERT+GraphCodeBert+UniXcoder\\GraphCodeBERT\\clonedetection\\parser\\tree-sitter-java',
    'F:\\PycharmProject\\CodeBERT+GraphCodeBert+UniXcoder\\GraphCodeBERT\\clonedetection\\parser\\tree-sitter-ruby',
    'F:\\PycharmProject\\CodeBERT+GraphCodeBert+UniXcoder\\GraphCodeBERT\\clonedetection\\parser\\tree-sitter-c-sharp',
  ]
)

