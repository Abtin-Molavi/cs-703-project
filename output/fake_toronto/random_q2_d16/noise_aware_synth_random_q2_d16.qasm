OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.1895541) q[8];
rz(0.74063263) q[11];
rz(3.1306224) q[11];
rz(3.7007457) q[11];
rz(5.9615689) q[11];
rz(3.5873849) q[11];
rz(3.5808868) q[11];
rz(5.3706793) q[11];
rz(2.854004) q[8];
rz(2.990635) q[8];
rz(3.7559486) q[11];
rz(0.11017289) q[11];
cx q[11],q[8];
rz(3.3344801) q[8];
rz(4.5295316) q[8];
rz(2.5133615) q[8];
rz(3.5070527) q[8];
rz(1.1042916) q[8];
rz(6.2708533) q[8];
rz(2.2169276) q[8];
rz(3.9192491) q[8];
cx q[11],q[8];
