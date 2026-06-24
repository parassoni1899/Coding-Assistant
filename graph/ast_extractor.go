// graph/ast_extractor.go — Native Go AST Parser
// ==============================================
// Compiles to a standalone binary that walks a Go repository, parses
// each .go file using the standard library's go/ast package, and emits
// a JSON array of Symbol objects on stdout.
//
// Build:
//     go build -o ast_extractor.exe ./graph/ast_extractor.go   (Windows)
//     go build -o ast_extractor     ./graph/ast_extractor.go   (Linux/Mac)
//
// Run:
//     ./ast_extractor <path-to-repo>
//
// Output:
//     JSON array → stdout
//     Errors     → stderr
//
// Design notes:
//   - Uses go/ast + go/token from the standard library — no external deps.
//   - Content is sliced from raw file bytes using Pos()/End() offsets,
//     which is more reliable than line-range re-reading.
//   - The binary is invoked by chunking/parser.py via subprocess.

package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// Symbol represents one extractable code unit from a Go source file.
type Symbol struct {
	Name      string   `json:"name"`
	Type      string   `json:"type"`     // "func" | "method" | "struct" | "interface"
	Receiver  string   `json:"receiver"` // empty for top-level funcs
	FilePath  string   `json:"file_path"`
	StartLine int      `json:"start_line"`
	EndLine   int      `json:"end_line"`
	Content   string   `json:"content"`
	Package   string   `json:"package"`
	Imports   []string `json:"imports"`
}

// shouldSkipDir returns true for directories that should never be traversed.
func shouldSkipDir(name string) bool {
	skip := map[string]bool{
		"vendor":       true,
		".git":         true,
		"testdata":     true,
		"node_modules": true,
	}
	return skip[name] || strings.HasPrefix(name, ".")
}

// shouldSkipFile returns true for generated or test files.
func shouldSkipFile(name string) bool {
	return strings.HasSuffix(name, "_test.go") ||
		strings.HasSuffix(name, ".pb.go") ||
		strings.HasSuffix(name, ".gen.go") ||
		strings.HasPrefix(name, "mock_")
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: ast_extractor <path-to-repo>")
		os.Exit(1)
	}

	repoPath := os.Args[1]
	fset := token.NewFileSet()
	var symbols []Symbol

	err := filepath.Walk(repoPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // skip inaccessible paths
		}

		// Prune skip directories
		if info.IsDir() {
			if shouldSkipDir(info.Name()) {
				return filepath.SkipDir
			}
			return nil
		}

		// Only process .go files
		if filepath.Ext(path) != ".go" {
			return nil
		}

		if shouldSkipFile(info.Name()) {
			return nil
		}

		// Read raw bytes (needed for content slicing)
		fileBytes, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "WARN: cannot read %s: %v\n", path, err)
			return nil
		}

		// Parse AST (including comments so doc-strings are part of the token range)
		fileAST, err := parser.ParseFile(fset, path, fileBytes, parser.ParseComments)
		if err != nil {
			// Many generated files have parse errors — just skip them
			return nil
		}

		pkgName := fileAST.Name.Name

		// Collect import paths
		var fileImports []string
		for _, imp := range fileAST.Imports {
			if imp.Path != nil {
				// imp.Path.Value is a quoted string literal — strip quotes
				importPath := strings.Trim(imp.Path.Value, `"`)
				fileImports = append(fileImports, importPath)
			}
		}
		if fileImports == nil {
			fileImports = []string{}
		}

		// Walk top-level declarations
		for _, decl := range fileAST.Decls {
			switch d := decl.(type) {

			case *ast.FuncDecl:
				// Slice the raw bytes to get the full function source
				startOffset := fset.Position(d.Pos()).Offset
				endOffset := fset.Position(d.End()).Offset
				if startOffset < 0 || endOffset > len(fileBytes) || startOffset >= endOffset {
					continue
				}
				content := string(fileBytes[startOffset:endOffset])

				sym := Symbol{
					Name:      d.Name.Name,
					Type:      "func",
					FilePath:  path,
					StartLine: fset.Position(d.Pos()).Line,
					EndLine:   fset.Position(d.End()).Line,
					Content:   content,
					Package:   pkgName,
					Imports:   fileImports,
				}

				// Methods have a receiver
				if d.Recv != nil && len(d.Recv.List) > 0 {
					sym.Type = "method"
					sym.Receiver = extractReceiver(d.Recv.List[0].Type)
				}

				symbols = append(symbols, sym)

			case *ast.GenDecl:
				// GenDecl covers type declarations (struct, interface, type alias)
				for _, spec := range d.Specs {
					ts, ok := spec.(*ast.TypeSpec)
					if !ok {
						continue
					}

					startOffset := fset.Position(d.Pos()).Offset
					endOffset := fset.Position(ts.End()).Offset
					if startOffset < 0 || endOffset > len(fileBytes) || startOffset >= endOffset {
						continue
					}
					content := string(fileBytes[startOffset:endOffset])

					sym := Symbol{
						Name:      ts.Name.Name,
						FilePath:  path,
						StartLine: fset.Position(d.Pos()).Line,
						EndLine:   fset.Position(ts.End()).Line,
						Content:   content,
						Package:   pkgName,
						Imports:   fileImports,
					}

					switch ts.Type.(type) {
					case *ast.StructType:
						sym.Type = "struct"
					case *ast.InterfaceType:
						sym.Type = "interface"
					default:
						continue // skip type aliases for now
					}

					symbols = append(symbols, sym)
				}
			}
		}

		return nil
	})

	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR walking repo: %v\n", err)
		os.Exit(1)
	}

	output, err := json.MarshalIndent(symbols, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR marshalling JSON: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(string(output))
}

// extractReceiver returns a human-readable receiver type string.
// e.g.,  *ast.StarExpr → "*PocketBase",  *ast.Ident → "App"
func extractReceiver(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		if ident, ok := t.X.(*ast.Ident); ok {
			return "*" + ident.Name
		}
	case *ast.IndexExpr:
		// Generic receivers: App[T]
		if ident, ok := t.X.(*ast.Ident); ok {
			return ident.Name
		}
	}
	return ""
}
