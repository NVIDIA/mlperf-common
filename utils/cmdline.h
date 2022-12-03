///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////////////
#ifndef INCLUDE_GUARD_MLPERF_UTIL_CMDLINE
#define INCLUDE_GUARD_MLPERF_UTIL_CMDLINE

#include <string>
#include <iostream>             // for std::cout
#include <sstream>              // for std::stringstream
#include <cassert>

///////////////////////////////////////////////////////////////////////////////

namespace cmdline {

///////////////////////////////////////////////////////////////////////////////
// Return true iff the string in [nameBegin, nameEnd) is matched by the "glob"
// pattern in [globBegin, globEnd).  Supports only '*' (any string) and '?'
// (any single char), patterns.  There is currently no way to escape '*' or '?'
// in the pattern.
///////////////////////////////////////////////////////////////////////////////
bool
globMatch(std::string::const_iterator nameBegin,
          std::string::const_iterator nameEnd,
          std::string::const_iterator globBegin,
          std::string::const_iterator globEnd);

///////////////////////////////////////////////////////////////////////////////
// Return true iff the string in [nameBegin, nameEnd) is matched by one or more
// "glob" patterns in the ':' separated list [globListBegin, globListEnd).
///////////////////////////////////////////////////////////////////////////////
bool
globListMatch(std::string::const_iterator nameBegin,
              std::string::const_iterator nameEnd,
              std::string::const_iterator globListBegin,
              std::string::const_iterator globListEnd);

// Returns index of first unprocessed arg (which might be argc if all the args
// were processed).
// Returns negative on error
// 
int
processArgs(int argc, char* argv[]);

extern std::string& helpMsg();

// the parts that must be virtualized rather than templatized:
class ParamDescrip
{
public:
    ParamDescrip(const std::string& shortname,
                 const std::string& longname,
                 const std::string& helpString);
    virtual ~ParamDescrip();
    // ParamDescrips are used by processArgs to fill in the args
    friend int processArgs(int argc, char* argv[]);
    virtual bool hasArg() const = 0;
    virtual int setValue(const std::string& inArg) = 0;
};

template< typename T >
class Param : protected ParamDescrip
{
    bool is_set;
    T value;
public:
    inline Param(const std::string& shortname,
                 const std::string& longname = "",
                 const std::string& helpString = "",
                 const T& dflt = T())
        : ParamDescrip(shortname, longname, helpString),
          is_set(false),
          value(dflt)
    {
    }
    // make this variable appear to be a "T" in any context where a "T" would
    // work.
    inline operator T() const { return value; }
    inline bool isSet() const { return is_set; }

protected:
    inline virtual bool hasArg() const { return true; }
    inline virtual int setValue(const std::string& inArg) {
        std::stringstream ss(inArg);
        // FIXME: Need error checking!  Check that this consumes the entirety
        // of ss without error.
        ss >> value;
        is_set = true;
        return 0;
    }
};

// specialization for bool (flag), which has no arg
template<>
class Param<bool> : protected ParamDescrip
{
    bool is_set;
    bool value;
public:
    inline Param<bool>(const std::string& shortname,
                       const std::string& longname = "",
                       const std::string& helpString = "",
                       const bool dflt = false)
    : ParamDescrip(shortname, longname, helpString),
      is_set(false),
      value(dflt)
    {
    }
    inline virtual bool hasArg() const { return false; }
    inline virtual int setValue(const std::string& inArg) {
        assert(inArg.length() == 0);
        value = !value;
        is_set = true;
        return 0;
    }
    inline operator bool() const { return value; }
    inline bool isSet() const { return is_set; }
};
// specialization for string which doesn't parse its arg
template<>
class Param<std::string> : protected ParamDescrip
{
    bool is_set;
    std::string value;
public:
    inline Param<std::string>(const std::string& shortname,
                              const std::string& longname = "",
                              const std::string& helpString = "",
                              const std::string& dflt = "")
    : ParamDescrip(shortname, longname, helpString),
      is_set(false),
      value(dflt)
    {
    }
    inline virtual bool hasArg() const { return true; }
    inline virtual int setValue(const std::string& inArg) {
        value = inArg;
        is_set = true;
        return 0;
    }
    // implicit casting won't help make this work with stream output because
    // stream operator<< is a template for strings, and implicit conversions
    // aren't considered.  https://stackoverflow.com/a/13883162/2209313
    inline operator const std::string&() const { return value; }
    inline bool isSet() const { return is_set; }
    inline std::string getString() const { return value; }
    // so we need to define the output operator
    friend std::ostream& operator<<(std::ostream& stream,
                                    Param<std::string> const& s);
    // and also begin/end operators
    inline std::string::const_iterator begin() const { return value.begin(); }
    inline std::string::const_iterator end() const { return value.end(); }
};

std::ostream& operator<<(std::ostream& stream,
                         Param<std::string> const& s);

///////////////////////////////////////////////////////////////////////////////

} // end namespace cmdline

#endif // INCLUDE_GUARD_MLPERF_UTIL_CMDLINE
