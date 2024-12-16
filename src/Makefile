###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

prefix      ?= /usr/local
exec_prefix ?= $(prefix)
bindir      ?= $(exec_prefix)/bin

PROGRAMS = alloc-empty-file-buffer cp-into fastcp

alloc_empty_OBJS = alloc-empty-file-buffer.o cmdline.o
cp_into_OBJS     = cp-into.o cmdline.o
fastcp_OBJS      = fastcp.o cmdline.o

LDFLAGS  = -s
CXXFLAGS = -O3 -Wall
LDLIBS   = -lpthread

.PHONY: all
all: $(PROGRAMS)

installdirs:
	mkdir --parents $(bindir)

install: all installdirs
	install -t $(bindir) $(PROGRAMS)

$(alloc_empty_OBJS) $(cp_into_OBJS) $(fastcp_OBJS): cmdline.h

alloc-empty-file-buffer: $(alloc_empty_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

cp-into: $(cp_into_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

fastcp: $(fastcp_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	$(RM) $(PROGRAMS) $(alloc_empty_OBJS) $(cp_into_OBJS) $(fastcp_OBJS)
