OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(4.901253) q[8];
rz(0.54093528) q[11];
rz(3.0274815) q[11];
rz(1.2017439) q[11];
rz(6.1477657) q[11];
rz(2.2261368) q[11];
rz(3.5231617) q[11];
rz(3.7120078) q[14];
rz(1.5094849) q[16];
cx q[16],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[16];
rz(0.7042347) q[16];
rz(0.71350104) q[16];
rz(0.54075859) q[16];
rz(2.2158505) q[16];
rz(4.6562528) q[16];
rz(1.0033839) q[16];
rz(1.0754204) q[16];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
rz(3.4182651) q[11];
rz(2.9895602) q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[11],q[14];
cx q[11],q[8];
cx q[8],q[11];
rz(3.107316) q[11];
rz(4.5529028) q[11];
rz(5.0352815) q[11];
rz(3.6715963) q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(1.4459097) q[8];
cx q[11],q[8];
rz(6.242467) q[11];
rz(2.0586653) q[11];
rz(2.5708871) q[8];
