#include "Parser.h"

void cmdParser::add_option(const std::string& key, const std::string& description)
{
  options.insert(std::make_pair(key, description));
}

void cmdParser::parse(int argc, char* argv[])
{
  for (int n=1; n<argc; n++){
    std::string key, val;
    if (argv[n][0] == '-' && ((argv[n][1]>='A' && argv[n][1]<='Z') || (argv[n][1]>='a' && argv[n][1]<='z')) )
    {
      if (n == argc-1){
        key = argv[n]+1;
        val = "no-arg";
      } else if (argv[n+1][0] == '-' && ((argv[n+1][1]>='A' && argv[n+1][1]<='Z') || (argv[n+1][1]>='a' && argv[n+1][1]<='z')) ) {
        key = argv[n]+1;
        val = "no-arg";
      } else if ((argv[n+1][0] != '-' || (argv[n+1][0] == '-' && argv[n+1][1] != '-'))) {
        key = argv[n]+1;
        val = argv[n+1];
        n++;
      }

    }
    else {
      std::string key = argv[n];
      throw std::runtime_error("[CASL_ERROR]: invalid option syntax '" + key + "'.");
    }

    if (key != "help" && options.find(key) == options.end())
    {
      std::cout << "[CASL_WARNING]: option '" << key << "' does not exists in the database -- ignoring." << std::endl;
      continue;
    }

    buffer.insert(std::make_pair(key, val));
  }

  if (contains("help")){
    std::cout << std::endl << std::endl;
    std::cout << " -------------== CASL Options Database ==------------- " << std::endl << std::endl;
    std::cout << " List of available options:" << std::endl << std::endl;
    for (std::map<std::string, std::string>::const_iterator it = options.begin(); it != options.end(); ++it)
    {
      std::cout << "  -" << it->first << ": " << it->second << std::endl;
    }
    std::cout << " ----------------------------------------------------- " << std::endl << std::endl;

    exit(EXIT_SUCCESS);
  }
}

bool cmdParser::contains(const std::string& key)
{
  return (buffer.find(key) != buffer.end());
}
