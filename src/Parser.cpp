#include "Parser.h"
#include <petsc.h>

void cmdParser::add_option(const std::string& key, const std::string& description)
{
  options.insert(std::make_pair(key, description));
}

bool cmdParser::parse(int argc, char* argv[], const std::string &extra_info)
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
      throw std::runtime_error("[ERROR]: invalid option syntax '" + key + "'.");
    }

    if (key != "help" && options.find(key) == options.end())
    {
      PetscPrintf(MPI_COMM_WORLD, "[WARNING]: option '%s' does not exists in the database -- ignoring.\n", key.c_str());
      continue;
    }

    buffer.insert(std::make_pair(key, val));
  }
  bool to_return = contains("help");

  if (to_return){
    if(!extra_info.empty())
    {
      PetscPrintf(MPI_COMM_WORLD, "\n-----------------------------== EXTRA INFORMATION FROM THE DEVELOPER ==----------------------------- \n\n");
      PetscPrintf(MPI_COMM_WORLD, extra_info.c_str());
    }
    PetscPrintf(MPI_COMM_WORLD, "\n ------------------== Available CASL Options ==------------------- \n\n");
    for (std::map<std::string, std::string>::const_iterator it = options.begin(); it != options.end(); ++it)
    {
      PetscPrintf(MPI_COMM_WORLD, "  -%s: %s\n", it->first.c_str(), it->second.c_str());
    }
    PetscPrintf(MPI_COMM_WORLD, " ----------------------------------------------------------------- \n\n");
  }
  else
    print();
  return to_return;
}

void cmdParser::print(FILE *f){
  PetscFPrintf(MPI_COMM_WORLD, f, " -------------------== CASL Options Database ==------------------- \n");
  PetscFPrintf(MPI_COMM_WORLD, f, " List of entered options:\n\n");
  for (std::map<std::string, std::string>::const_iterator it = buffer.begin(); it != buffer.end(); ++it)
    PetscFPrintf(MPI_COMM_WORLD, f, "  -%s %s\n", it->first.c_str(), it->second.c_str());
  PetscPrintf(MPI_COMM_WORLD, " ----------------------------------------------------------------- \n");    
}


bool cmdParser::contains(const std::string& key) const
{
  return (buffer.find(key) != buffer.end());
}
