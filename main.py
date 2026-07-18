import re
import sys
from pathlib import Path
from typing import Any, Union
import numpy as np


class MATPOWERCase(dict):
    """
    A dictionary-like object representing a MATPOWER case (mpc) that also
    supports attribute access (e.g., mpc.bus, mpc.baseMVA, mpc.gen).
    """
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'MATPOWERCase' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'MATPOWERCase' object has no attribute '{key}'")

    def summary(self) -> str:
        """Returns a concise summary of the MATPOWER case."""
        lines = [f"MATPOWER Case Summary (version: {self.get('version', 'N/A')})"]
        lines.append(f"  baseMVA: {self.get('baseMVA', 'N/A')}")
        for k, v in self.items():
            if isinstance(v, np.ndarray):
                lines.append(f"  {k}: {v.shape[0]} rows x {v.shape[1]} columns ({v.dtype})")
            elif isinstance(v, list):
                lines.append(f"  {k}: list of {len(v)} items")
            elif k not in ('version', 'baseMVA'):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# --- Standard MATPOWER Column Definitions for convenience ---
# Bus columns
BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
GS = 4
BS = 5
BUS_AREA = 6
VM = 7
VA = 8
BASE_KV = 9
ZONE = 10
VMAX = 11
VMIN = 12

# Generator columns
GEN_BUS = 0
PG = 1
QG = 2
QMAX = 3
QMIN = 4
VG = 5
MBASE = 6
GEN_STATUS = 7
PMAX = 8
PMIN = 9

# Branch columns
F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
RATE_A = 5
RATE_B = 6
RATE_C = 7
TAP = 8
SHIFT = 9
BR_STATUS = 10
ANGMIN = 11
ANGMAX = 12


def parse_matpower_string(content: str) -> MATPOWERCase:
    """
    Parses a MATPOWER format string (e.g. from an .m case file like ACTIVSg2000.m)
    into a MATPOWERCase object containing numpy arrays for bus, gen, branch, etc.
    """
    mpc = MATPOWERCase()

    current_matrix_field = None
    current_matrix_rows = []

    current_cell_field = None
    current_cell_items = []

    for line in content.splitlines():
        # Strip MATLAB/Python style comments (% or #) outside of string literals
        # For simplicity in standard MATPOWER files, % after data or at start of line is a comment.
        line_clean = re.split(r'[%#]', line, maxsplit=1)[0].strip()
        if not line_clean:
            continue

        # 1. If currently parsing a multi-line matrix [ ... ]
        if current_matrix_field is not None:
            if ']' in line_clean:
                data_part, rest = line_clean.split(']', 1)
                _parse_matrix_rows(data_part, current_matrix_rows)
                # Convert collected rows to a 2D numpy array of floats
                if current_matrix_rows:
                    mpc[current_matrix_field] = np.array(current_matrix_rows, dtype=float)
                else:
                    mpc[current_matrix_field] = np.empty((0, 0), dtype=float)
                current_matrix_field = None
                line_clean = rest.strip()
                if not line_clean or line_clean == ';':
                    continue
            else:
                _parse_matrix_rows(line_clean, current_matrix_rows)
                continue

        # 2. If currently parsing a multi-line cell array { ... }
        if current_cell_field is not None:
            if '}' in line_clean:
                data_part, rest = line_clean.split('}', 1)
                _parse_cell_items(data_part, current_cell_items)
                mpc[current_cell_field] = current_cell_items
                current_cell_field = None
                line_clean = rest.strip()
                if not line_clean or line_clean == ';':
                    continue
            else:
                _parse_cell_items(line_clean, current_cell_items)
                continue

        # 3. Check for new assignments of the form: mpc.<field> = ...
        match = re.match(r'(?:[a-zA-Z0-9_]+\.)?([a-zA-Z0-9_]+)\s*=\s*(.*)', line_clean)
        if not match:
            continue

        field = match.group(1)
        rhs = match.group(2).strip()

        # Check if RHS starts with matrix '['
        if rhs.startswith('['):
            rhs_inner = rhs[1:].strip()
            if ']' in rhs_inner:
                data_part, _ = rhs_inner.split(']', 1)
                rows = []
                _parse_matrix_rows(data_part, rows)
                mpc[field] = np.array(rows, dtype=float) if rows else np.empty((0, 0), dtype=float)
            else:
                current_matrix_field = field
                current_matrix_rows = []
                _parse_matrix_rows(rhs_inner, current_matrix_rows)
            continue

        # Check if RHS starts with cell array '{'
        if rhs.startswith('{'):
            rhs_inner = rhs[1:].strip()
            if '}' in rhs_inner:
                data_part, _ = rhs_inner.split('}', 1)
                items = []
                _parse_cell_items(data_part, items)
                mpc[field] = items
            else:
                current_cell_field = field
                current_cell_items = []
                _parse_cell_items(rhs_inner, current_cell_items)
            continue

        # Otherwise, scalar/string assignment: e.g., '2'; or 100.0;
        rhs_clean = rhs.rstrip(';').strip()
        if (rhs_clean.startswith("'") and rhs_clean.endswith("'")) or \
           (rhs_clean.startswith('"') and rhs_clean.endswith('"')):
            mpc[field] = rhs_clean[1:-1]
        else:
            try:
                # Try converting to int or float if numeric
                if '.' in rhs_clean or 'e' in rhs_clean.lower():
                    mpc[field] = float(rhs_clean)
                else:
                    mpc[field] = int(rhs_clean)
            except ValueError:
                mpc[field] = rhs_clean

    return mpc


def _parse_matrix_rows(text: str, rows: list) -> None:
    """Helper to parse rows separated by ';' from text and append to rows list."""
    for row_str in text.split(';'):
        row_str = row_str.strip()
        if not row_str:
            continue
        # Split on whitespace and/or commas
        tokens = re.split(r'[\s,]+', row_str)
        try:
            row_vals = [float(token) for token in tokens if token]
            if row_vals:
                rows.append(row_vals)
        except ValueError:
            # If tokens cannot be converted to float (e.g. non-numeric strings), skip or handle
            pass


def _parse_cell_items(text: str, items: list) -> None:
    """Helper to parse string literals inside cell arrays separated by ';' or ','."""
    matches = re.findall(r'["\']([^"\']*)["\']', text)
    items.extend(matches)


def load_matpower_case(filepath: Union[str, Path]) -> MATPOWERCase:
    """Reads and parses a MATPOWER .m case file from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Case file not found: {filepath}")
    return parse_matpower_string(path.read_text(encoding='utf-8'))


def check_bus_loads(mpc: MATPOWERCase) -> dict[str, float]:
    """
    Checks and summarizes the total active (Pd) and reactive (Qd) power demand
    in the buses matrix across all buses and by bus type (PQ, PV, Ref, Isolated).
    """
    if 'bus' not in mpc or mpc.bus.size == 0:
        print("No bus data found in MATPOWER case.")
        return {}

    bus = mpc.bus
    types = bus[:, BUS_TYPE].astype(int)
    pd_vals = bus[:, PD]
    qd_vals = bus[:, QD]

    type_names = {
        1: "PQ (Load)",
        2: "PV (Gen)",
        3: "Ref (Slack)",
        4: "Isolated"
    }

    print("\n" + "=" * 62)
    print(f"{'BUS LOAD CHECK (Pd & Qd Summary)':^62}")
    print("=" * 62)
    print(f"{'Bus Type':<18} | {'Count':>7} | {'Total Pd (MW)':>14} | {'Total Qd (MVAr)':>15}")
    print("-" * 62)

    results = {}
    total_pd = float(np.sum(pd_vals))
    total_qd = float(np.sum(qd_vals))

    for b_type in [1, 2, 3, 4]:
        mask = (types == b_type)
        count = int(np.sum(mask))
        if count > 0:
            pd_sum = float(np.sum(pd_vals[mask]))
            qd_sum = float(np.sum(qd_vals[mask]))
            t_name = type_names.get(b_type, f"Type {b_type}")
            print(f"{t_name:<18} | {count:>7,d} | {pd_sum:>14.4f} | {qd_sum:>15.4f}")
            results[f"Pd_type_{b_type}"] = pd_sum
            results[f"Qd_type_{b_type}"] = qd_sum

    print("-" * 62)
    print(f"{'TOTAL (All Buses)':<18} | {len(bus):>7,d} | {total_pd:>14.4f} | {total_qd:>15.4f}")
    print("=" * 62)

    # Additional diagnostic check on how many buses actually have non-zero demand
    nonzero_mask = (pd_vals != 0.0) | (qd_vals != 0.0)
    nonzero_count = int(np.sum(nonzero_mask))
    print(f"-> Buses with non-zero load (Pd != 0 or Qd != 0): {nonzero_count,d} / {len(bus),d}")
    if nonzero_count == 0:
        print("-> NOTE: All buses have Pd = 0.0 and Qd = 0.0 in this case file.")
    print("=" * 62 + "\n")

    results["total_Pd"] = total_pd
    results["total_Qd"] = total_qd
    return results



if __name__ == "__main__":
    # If a file path is provided via command-line (e.g. `python main.py ACTIVSg2000.m`), load from that file.
    if len(sys.argv) > 1:
        case_path = sys.argv[1]
        print(f"--- Loading MATPOWER case from file: {case_path} ---")
        mpc = load_matpower_case(case_path)
    else:
        # Check if any .m file exists in the current directory and load the first one
        example_files = list(Path('.').glob('*.m'))
        if example_files:
            case_path = example_files[0]
            print(f"--- Loading MATPOWER case from file: {case_path} ---")
            mpc = load_matpower_case(case_path)
        else:
            print("Usage: python main.py <path_to_case_file.m>")
            print("No .m file passed via command line or found in the current directory.")
            print("Running fallback demo using in-memory ACTIVSg2000 snippet instead:\n")
            sample_activsg2000 = """
function mpc = ACTIVSg2000
mpc.version = '2';
mpc.baseMVA = 100.0;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1001	1	0.0	0.0	0.0	0.0	1	1.0	0.0	115.0	9	1.5	0.5	; % TopologicalNode 938c5fa9-eabc-59bf-988a-c1bb8dff3f22 "ODESSA 2 0"
	1002	1	25.0	12.0	0.0	0.0	1	0.99	-1.2	115.0	9	1.5	0.5	; % Bus 2
	1003	2	15.0	8.0	0.0	0.0	1	1.02	-0.5	115.0	9	1.5	0.5	; % Bus 3
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin
mpc.gen = [
	1003	50.0	10.0	100.0	-50.0	1.02	100.0	1	200.0	0.0	;
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1001	1002	0.01	0.05	0.02	100	100	100	0	0	1	-360	360	;
	1002	1003	0.015	0.06	0.025	100	100	100	0	0	1	-360	360	;
];
"""
    mpc = load_matpower_case("/usr/local/google/home/sxzhou/Downloads/activsg.m")

    print(mpc.summary())
    print("\nAccessing fields using dot notation:")
    print(f"Version: {mpc.get('version')}")
    print(f"Base MVA: {mpc.get('baseMVA')}")
    if 'bus' in mpc:
        print(f"\nBus matrix shape: {mpc.bus.shape}")
        print("Bus IDs (first 10):", mpc.bus[:10, BUS_I].astype(int))
        print("Bus Base KV (first 10):", mpc.bus[:10, BASE_KV])
        
        # Run detailed Pd and Qd (PQ) bus load check
        check_bus_loads(mpc)
