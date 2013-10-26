#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <map>
#include <stdexcept>

class cmdParser{
    std::map <std::string, std::string> options;
    std::map <std::string, std::string> buffer;

public:   
    /*!
     * \brief add_option: Adds a new option into the database
     * \param key: option name. This what you enter at the command-line
     * \param description: Description of this option. This will be displayed if user runs with -help
     */
    void add_option(const std::string& key, const std::string& description);

    /*!
     * \brief parse: parses the input and genrates the option database
     * \param argc: argc parameter from main function
     * \param argv: argc parameter from main function
     */
    void parse(int argc, char* argv[]);

    /*!
     * \brief contains: searches the option database for a specific key
     * \param key: key to search for
     * \return returns true if the option exists in the database and entered by the user
     */
    bool contains(const std::string& key);

    /*!
     * \brief get: returns the value of the option asked for
     * \param key: the key to search the value for
     * \return returns the value of teh option
     * \note this function uses stringstream objects. As long as you overload the stream operators, you can use it with any object you define
     */
    template<typename T>
    T get(const std::string& key)
    {
        if (options.find(key) == options.end())
            throw std::runtime_error("[CASL_ERROR]: Option '" + key + "' was not found in option database.");
        else if (options.find(key) != options.end() && buffer.find(key) == buffer.end())
            throw std::runtime_error("[CASL_ERROR]: Option '" + key + "' was found in option database but was not entered.");
        else
        {
            if (!buffer[key].compare("no-arg"))
                throw std::runtime_error("[CASL_ERROR]: option '" + key + "' does not include any value");
            std::istringstream iss(buffer[key]);
            T tmp; iss >> tmp;
            return tmp;
        }
    }

    /*!
     * \brief get: same as before, except now will return the default value if the user did not provide with a value at the command line
     * \param key: the key to search the value for
     * \param default_value: use this value if the user did not enter a value for the option
     * \return returns the value of teh option
     * \note this function uses stringstream objects. As long as you overload the stream operators, you can use it with any object you define
     */
    template<typename T>
    T get(const std::string& key, const T& default_value)
    {
        if (options.find(key) == options.end())
            return default_value;
        else if (options.find(key) != options.end() && buffer.find(key) == buffer.end())
            return default_value;
        else
        {
            if (!buffer[key].compare("no-arg"))
                return default_value;
            std::istringstream iss(buffer[key]);
            T tmp; iss >> tmp;
            return tmp;
        }
    }
};

#endif // PARSER_H
