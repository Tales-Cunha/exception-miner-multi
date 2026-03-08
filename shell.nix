{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [ 
    python310 
    stdenv.cc.cc.lib
    zlib
  ];

  shellHook = ''
    echo "Python 3.10 environment loaded"
    python --version
    
    # Set library path for compiled packages
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
      echo "Creating virtual environment..."
      python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    echo "Virtual environment activated. Run 'pip install -r requirements.txt' to install dependencies."
  '';
}