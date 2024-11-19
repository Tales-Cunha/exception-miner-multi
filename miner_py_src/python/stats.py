from collections import Counter
from .miner_py_utils import (
    count_except,
    count_raise,
    statement_couter,
    is_try_except_pass,
    is_generic_except,
    count_broad_exception_raised,
    count_try_except_raise,
    count_misplaced_bare_raise,
    count_try_else,
    count_try_return,
    count_finally,
    get_except_identifiers,
    get_raise_identifiers,
    get_except_clause,
    get_except_block,
    count_nested_try,
    is_bare_except,
    count_bare_raise_inside_finally,
    get_code_without_try_except,
    get_try_statements_vector
)
from tqdm import tqdm
from tree_sitter._binding import Node, Tree
from .miner_py_utils import QUERY_TRY_STMT, QUERY_EXCEPT_CLAUSE


class FileStats:
    num_files = 0
    num_functions = 0
    files_try_except = set()
    files_try_pass = set()
    files_generic_except = set()
    func_try_except = set()
    func_try_pass = set()
    func_generic_except = set()
    func_has_except_handler = set()
    func_has_nested_try = set()

    def metrics(self, func_def: Node, file_path: str):
        if len(QUERY_TRY_STMT.captures(func_def)) != 0:
            self.files_try_except.add(file_path)

        captures_except = QUERY_EXCEPT_CLAUSE.captures(func_def)

        for except_clause, _ in captures_except:
            self.func_try_except.add(f"{file_path}:{func_def.id}")
            if is_try_except_pass(except_clause):
                self.func_try_pass.add(f"{file_path}:{func_def.id}")
                self.files_try_pass.add(file_path)
            if is_generic_except(except_clause):
                tqdm.write(f"{file_path}:{func_def.id}")
                self.files_generic_except.add(file_path)
                self.func_generic_except.add(f"{file_path}:{func_def.id}")

    def __str__(self) -> str:
        return (
            f"\n---------------- try-except STATS -----------------\n"
            f"# try-except found:                   {len(self.func_try_except)}\n"
            f"% try-except per file:                {(len(self.files_try_except) / self.num_files) * 100:.2f}%\n"
            f"% try-except per function definition: {(len(self.func_try_except) / max(self.num_functions, 1)) * 100:.2f}%\n"
            f"\n--------------- bad practice STATS ----------------\n"
            f"# try-pass:                           {len(self.func_try_pass)}\n"
            f"% try-pass per file:                  {(len(self.files_try_pass) / self.num_files) * 100:.2f}%\n"
            f"% try-pass per function definition:   {(len(self.func_try_pass) / max(self.num_functions, 1)) * 100:.2f}%\n"
            f"# generic exception handling:         {len(self.func_generic_except)}\n"
            f"# generic exception per file:         {(len(self.files_generic_except) / self.num_files) * 100:.2f}%\n"
            f"# generic exception per function definition: {(len(self.func_generic_except) / max(self.num_functions, 1)) * 100:.2f}%\n"
        )

    def get_metrics(self, func_def: Node, tree: None):
        """
        Return a list of exception handling metrics in the following order: try-except clauses,
            try-pass, generic-except
        """
        n_try_except, n_try_pass, n_generic_except, n_bare_except = 0, 0, 0, 0

        captures_except = get_except_clause(func_def)

        n_captures_broad_raise = count_broad_exception_raised(func_def)

        n_captures_try_except_raise = count_try_except_raise(func_def)

        n_captures_misplaced_bare_raise = count_misplaced_bare_raise(func_def)

        n_bare_raise_finally = count_bare_raise_inside_finally(func_def)

        n_raise = count_raise(func_def)

        n_try_else = count_try_else(func_def)

        n_try_return = count_try_return(func_def)

        #captures_except_ident = QUERY_EXCEPT_IDENTIFIER.captures(func_def)       

        n_nested_try = count_nested_try(func_def)

        n_finally = count_finally(func_def) 
    
        captures_except_ident = get_except_identifiers(func_def)

        captures_raise_ident = get_raise_identifiers(func_def)

        captures_except_block = list(map(lambda x: x[0].text.decode('utf-8'), get_except_block(func_def)))

        captures_except_clause = list(map(lambda x: x[0].text.decode('utf-8'), get_except_clause(func_def)))

        captures_code_without_try_except = get_code_without_try_except(func_def, tree)

        captures_try_stmt_vec = get_try_statements_vector(func_def)
        
        for except_clause, _ in captures_except:
            n_try_except += 1
            if is_try_except_pass(except_clause):
                n_try_pass += 1
            if is_generic_except(except_clause):
                # tqdm.write(f"{file_path}:{func_def.id}")
                n_generic_except += 1
            if is_bare_except(except_clause):
                n_bare_except += 1

        str_except_identifiers = " ".join(captures_except_ident)
        str_raise_identifiers = " ".join(captures_raise_ident)
        str_except_block = " ".join(captures_except_block)

        return {
            "n_try_except": n_try_except,
            "n_try_pass": n_try_pass,
            "n_finally": n_finally,
            "n_generic_except": n_generic_except,
            "n_raise": n_raise,
            "n_captures_broad_raise": n_captures_broad_raise,
            "n_captures_try_except_raise": n_captures_try_except_raise,
            "n_captures_misplaced_bare_raise": n_captures_misplaced_bare_raise,
            "n_try_else": n_try_else,
            "n_try_return": n_try_return,
            "str_except_identifiers": str_except_identifiers,
            "str_raise_identifiers": str_raise_identifiers,
            "str_except_block": str_except_block,
            "n_nested_try": n_nested_try,
            "n_bare_except": n_bare_except,
            "n_bare_raise_finally" : n_bare_raise_finally,
            "str_code_without_try_except": captures_code_without_try_except,
            "str_captures_except": captures_except_clause
        }


class TBLDStats:
    functions_count = 0
    try_num_eq_1 = 0
    try_num_lt_eq_2 = 0
    tokens_count = 0
    num_max_tokens = 0
    statements_count = 0
    num_max_statement = 0
    unique_tokens = Counter()
    function_tokens_acc = 0

    def increment_try_stats(self, try_count):
        if try_count == 1:
            self.try_num_eq_1 += 1
        elif try_count >= 2:
            self.try_num_lt_eq_2 += 1

    def __str__(self) -> str:
        stats_str = (
            "-------- TBLD STATS --------\n"
            f"#Python methods   {self.functions_count}\n"
            f"#TryNum=1         {self.try_num_eq_1}\n"
            f"#TryNum≥2         {self.try_num_lt_eq_2}\n"
            f"#MaxTokens        {self.num_max_tokens}\n"
            f"#AverageTokens    {self.tokens_count / self.functions_count if self.functions_count != 0 else 0:.2f}\n"
            f"#MaxStatements    {self.num_max_statement}\n"
            f"#AverageStatement {(self.statements_count / self.functions_count) if self.functions_count != 0 else 0:.2f}\n"
            f"#UniqueTokens     {len(self.unique_tokens)}\n"
            "-------- Top 10 Unique Tokens Ranking --------\n"
        )

        unique_ranking_str = "".join(
            [
                f"{key} - {value}" + "\n"
                for key, value in self.unique_tokens.most_common()[:10]
            ]
        )
        return stats_str + unique_ranking_str


class CBGDStats:
    functions_count = 0
    except_num_eq_1 = 0
    except_num_lt_eq_2 = 0
    tokens_count_source = 0
    tokens_count_target = 0
    num_max_tokens_source = 0
    num_max_tokens_target = 0
    statements_count = 0
    num_max_statement = 0
    unique_tokens = Counter()
    function_tokens_source_acc = 0
    function_tokens_target_acc = 0

    current_num_tokens = 0

    def reset(self):
        self.num_max_tokens_source = max(
            self.num_max_tokens_source, self.function_tokens_source_acc
        )
        self.tokens_count_source += self.function_tokens_source_acc
        self.function_tokens_source_acc = 0

        self.num_max_tokens_target = max(
            self.num_max_tokens_target, self.function_tokens_target_acc
        )
        self.tokens_count_target += self.function_tokens_target_acc
        self.function_tokens_target_acc = 0

    def increment_except_stats(self, function_def: Node):
        except_count = count_except(function_def)
        if except_count == 1:
            self.except_num_eq_1 += 1
        elif except_count >= 2:
            self.except_num_lt_eq_2 += 1

    def increment_function_counter(self):
        self.functions_count += 1

    def increment_statements_counter(self, function_def: Node):
        num_statements = statement_couter(function_def)
        self.statements_count += num_statements
        self.num_max_statement = max(self.num_max_statement, num_statements)

    def increment_current_num_tokens(self, num_tokens):
        self.current_num_tokens = num_tokens

    def move_current_num_tokens_source(self):
        self.function_tokens_source_acc += self.current_num_tokens
        self.current_num_tokens = 0

    def move_current_num_tokens_target(self):
        self.function_tokens_target_acc += self.current_num_tokens
        self.current_num_tokens = 0

    def __str__(self) -> str:
        stats_str = (
            "-------- CBGD STATS --------\n"
            f"#Try-catch pairs         {self.functions_count}\n"
            f"#ExceptNum=1             {self.except_num_eq_1}\n"
            f"#ExceptNum≥2             {self.except_num_lt_eq_2}\n"
            f"#MaxTokens of Source     {self.num_max_tokens_source}\n"
            f"#AverageTokens of Source {self.tokens_count_source / self.functions_count if self.functions_count != 0 else 0:.2f}\n"
            f"#MaxTokens of Target     {self.num_max_tokens_target}\n"
            f"#AverageTokens of Target {self.tokens_count_target / self.functions_count if self.functions_count != 0 else 0:.2f}\n"
            f"#UniqueTokens            {len(self.unique_tokens)}\n"
            "-------- Top 10 Unique Tokens Ranking --------\n"
        )

        unique_ranking_str = "".join(
            [
                f"{key} - {value}" + "\n"
                for key, value in self.unique_tokens.most_common()[:10]
            ]
        )
        return stats_str + unique_ranking_str
