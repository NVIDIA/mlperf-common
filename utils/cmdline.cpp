
#include "cmdline.h"

#include <string>
#include <algorithm>
#include <map>
#include <cassert>

///////////////////////////////////////////////////////////////////////////////

namespace cmdline {

///////////////////////////////////////////////////////////////////////////////

bool
globMatch(std::string::const_iterator nameBegin,
          std::string::const_iterator nameEnd,
          std::string::const_iterator globBegin,
          std::string::const_iterator globEnd)
{
    // base case: empty pattern
    if( globBegin == globEnd ) {
        // empty pattern matches only empty string
        return nameBegin == nameEnd;
    }
    switch( *globBegin ) {
        case '?' : // match any non empty char
            return ((nameBegin != nameEnd) &&
                    globMatch(nameBegin+1, nameEnd,
                              globBegin+1, globEnd));
        case '*' : // match any string (including empty string)
            return (((nameBegin != nameEnd) &&
                     globMatch(nameBegin+1, nameEnd,
                               globBegin, globEnd)) ||
                    globMatch(nameBegin, nameEnd,
                              globBegin+1, globEnd));
        default  :              // match next literal
            return ((*nameBegin == *globBegin) &&
                    globMatch(nameBegin+1, nameEnd,
                              globBegin+1, globEnd));
    }
}

///////////////////////////////////////////////////////////////////////////////

bool
globListMatch(std::string::const_iterator nameBegin,
              std::string::const_iterator nameEnd,
              std::string::const_iterator globListBegin,
              std::string::const_iterator globListEnd)
{
    auto globBegin = globListBegin;
    do {
        // find() returns globListEnd if it can't find ':'
        auto globEnd = std::find(globBegin, globListEnd, ':');
        if( globMatch(nameBegin, nameEnd,
                      globBegin, globEnd) ) {
            return true;
        }

        if( globEnd == globListEnd ) {
            // that was the last one :(
            return false;
        }
        // Move past the ':' char.  globEnd != globListEnd here so this is well
        // defined
        globBegin = globEnd + 1;
    } while( true );

    return false;
}

///////////////////////////////////////////////////////////////////////////////

class ParamDescrip;
typedef std::map<const std::string, ParamDescrip*> ParamList;

static ParamList& paramList() {
    static ParamList* paramListPtr = 0;
    if( paramListPtr == 0 ) {
        paramListPtr = new ParamList;
    }
    return *paramListPtr;
}

std::string& helpMsg() {
    static std::string* helpMsgPtr = 0;
    if( helpMsgPtr == 0 ) {
        helpMsgPtr = new std::string;
    }
    return *helpMsgPtr;
}

///////////////////////////////////////////////////////////////////////////////

int
processArgs(int   argc,
            char* argv[])
{
    int i = 1;                  // 
    for( ; i < argc; ++i ) {
        const std::string rawArg = argv[i];
        auto equalsPos = std::find(rawArg.begin(), rawArg.end(), '=');
        const std::string paramName(rawArg.begin(), equalsPos);
        bool hasEquals = equalsPos != rawArg.end();

        auto found = paramList().find(paramName);
        if( found == paramList().end() ) {
            return i;
        }
        ParamDescrip* param = found->second;
        if( param->hasArg() ) {
            if( hasEquals ) {
                // input is of form "--param=value", arg is "value" (could be
                // empty string)
                const std::string theArg(equalsPos+1, rawArg.end());
                param->setValue(theArg);
            } else {
                // input is of form "-param value"
                if( i+1 >= argc ) {
                    std::cerr << paramName << " requires an argument" << std::endl;
                    exit(1);
                }
                ++i;
                param->setValue(argv[i]);
            }
        } else {
            if( hasEquals ) {
                std::cerr << "Flag " << paramName <<
                    " does not take an argument" << std::endl;
                exit(1);
            }
            param->setValue(""); // set flag true
        }
    }
    return i;
}

///////////////////////////////////////////////////////////////////////////////

ParamDescrip::ParamDescrip(const std::string& shortname,
                           const std::string& longname,
                           const std::string& helpString)
{
    assert((shortname != "" || longname != "") && "Need a non-empty argument name!" );

    if( shortname != "" ) {
        paramList()[shortname] = this;
    }
    if( longname != "" ) {
        paramList()[longname] = this;
    }
    if( helpString != "" ) {
        if( shortname != "" ) {
            helpMsg() += shortname;
            if( longname != "" ) {
                helpMsg() += ", ";
            }
        }
        if( longname != "" ) {
            helpMsg() += longname;
        }
        helpMsg() += "\n";
        helpMsg() += "\t";
        helpMsg() += helpString;
        helpMsg() += "\n";
    }
}

///////////////////////////////////////////////////////////////////////////////

ParamDescrip::~ParamDescrip()
{
    // empty
}

///////////////////////////////////////////////////////////////////////////////

std::ostream&
operator<<(std::ostream& stream,
           Param<std::string> const& s)
{
    return stream << s.value;
}

///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

} // end namespace cmdline
